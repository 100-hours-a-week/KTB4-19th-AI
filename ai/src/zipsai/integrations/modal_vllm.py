import os
import time
import urllib.request

import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
N_GPU = 1
GPU_TYPE = "L4"
MINUTES = 60
VLLM_PORT = 8000

app = modal.App("zipsai-vllm-qwen3-vl")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # 모델 다운로드 가속
            "VLLM_SERVER_DEV_MODE": "1",  # /sleep, /wake_up 엔드포인트 등록(sleep mode 필수)
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # GPU 메모리 스냅샷과 torch.compile 호환성(공식 가이드 권장)
        }
    )
)

# 컨테이너가 꺼졌다 켜질 때마다 모델을 huggingface.co에서 새로 받지 않도록 캐시.
hf_cache_vol = modal.Volume.from_name("zipsai-hf-cache", create_if_missing=True)

vllm_api_key_secret = modal.Secret.from_name("zipsai-vllm-api-key")


def _vllm_request(method: str, path: str, timeout: float) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}{path}",
        method=method,
        headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def _wait_ready(deadline_seconds: float) -> None:
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        try:
            if _vllm_request("GET", "/v1/models", timeout=5) == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM이 제한 시간 안에 준비되지 않음")


@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=15 * MINUTES,  # 요청 없이 이 시간 지나면 컨테이너 내림(과금 중단)
    startup_timeout=10 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    port=VLLM_PORT,
    secrets=[vllm_api_key_secret],
    unauthenticated=True,  # Modal 프록시 인증 대신 vllm --api-key로 인증(기존 serverless 슬롯과 동일 방식)
    target_concurrency=100,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Server:
    @modal.enter(snap=True)
    def start(self):
        """스냅샷을 만들 때(=최초 1회) 돈다: vLLM을 끝까지 띄워서 torch.compile·CUDA graph
        capture·FlashInfer JIT이 전부 끝나게 한 뒤, GPU 메모리를 비우고(sleep) 그 상태를 찍는다."""
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--max-model-len",
            "8192",  # L4(24GiB)에서 가중치+KV cache가 들어가는 상한
            "--gpu-memory-utilization",
            "0.95",
            "--enable-sleep-mode",  # sleep()/wake_up() 가능하게. 스냅샷 전 GPU 비우기용
        ]
        key_index = cmd.index("--api-key") + 1
        print(*(cmd[:key_index] + ["***"] + cmd[key_index + 1 :]))
        self.process = subprocess.Popen(cmd)

        _wait_ready(10 * MINUTES)

        status = _vllm_request("POST", "/sleep?level=1", timeout=30)
        print(f"sleep level=1: HTTP {status}")

    @modal.enter(snap=False)
    def wake_up(self):
        """스냅샷 복원 때마다 돈다(최초 1회 포함): GPU로 가중치를 되돌려 깨운다."""
        status = _vllm_request("POST", "/wake_up", timeout=60)
        print(f"wake_up: HTTP {status}")
        _wait_ready(2 * MINUTES)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
def test():
    """modal run으로 실행: 콜드스타트 포함 헬스체크 1회 + 모델 목록 조회.

    이 함수는 로컬에서 실행되므로(컨테이너 안이 아님) VLLM_API_KEY를
    로컬 셸에도 동일하게 export 해둬야 함: export VLLM_API_KEY=<secret에 넣은 값>
    """
    url = Server.get_url()
    print(f"server url: {url}")

    start = time.time()
    deadline = start + 10 * MINUTES
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError("health check timed out")
    print(f"ready: {time.time() - start:.1f}s")

    req = urllib.request.Request(
        f"{url}/v1/models",
        headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(resp.read().decode())
