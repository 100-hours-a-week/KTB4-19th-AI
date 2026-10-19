"""로컬 이미지 한 장으로 VLM을 두 가지 방식으로 찔러본다 — 실제 네트워크 호출.

1) 자유 서술: 포맷 제약 없이 이미지를 읽을 수 있는지만 본다.
2) 프로덕션 프롬프트: VLM_ANALYSIS_PROMPT로 물어서 우리 출력 계약(_ModelImages)에
   맞는 JSON을 주는지 본다. 코드펜스로 감싸서 주면 여기서 FAIL로 나온다.

사용: uv run python scripts/check_vlm_image_analysis.py <이미지 경로>
"""

import base64
import mimetypes
import sys

from pydantic import ValidationError

from zipsai.complaint.prompts import VLM_ANALYSIS_PROMPT
from zipsai.integrations.llm import strip_json_code_fence
from zipsai.integrations.vlm import _get_client, _ModelImages
from zipsai.settings import get_settings


def _as_data_url(image_path: str) -> str:
    content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _ask(client, model: str, prompt: str, data_url: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def main(image_path: str) -> None:
    settings = get_settings()
    client = _get_client(
        settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_seconds
    )
    data_url = _as_data_url(image_path)

    print("=== 1) 자유 서술 — VLM이 이미지를 읽긴 하는가? ===")
    print(
        _ask(
            client, settings.vlm_model, "이 이미지에 뭐가 보이는지 설명해줘.", data_url
        )
    )

    print("\n=== 2) 프로덕션 프롬프트 — 우리 출력 계약을 지키는가? ===")
    raw = _ask(client, settings.vlm_model, VLM_ANALYSIS_PROMPT, data_url)
    print(raw)

    print("\n=== 결과 ===")
    try:
        parsed = _ModelImages.model_validate_json(strip_json_code_fence(raw))
        print("PASS:", parsed.model_dump())
    except ValidationError as error:
        print("FAIL:", error)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_vlm_image_analysis.py <image_path>")
    main(sys.argv[1])
