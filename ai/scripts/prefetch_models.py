"""빌드 단계에서 모델을 내려받아 이미지에 넣는다. 런타임에는 바깥으로 나가지 않는다.

받을 목록을 따로 적는 대신 실제 파싱 경로를 한 번 태운다 — 설정이 바뀌면 받는 모델도
같이 따라온다. 태우기만 하면 실패를 놓치므로 끝나고 캐시를 직접 확인한다.
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import rapidocr
from PIL import Image, ImageDraw

from zipsai.indexing.parse import OCR_LANGUAGE, _converter


def warm_easyocr(sample: Path) -> None:
    import easyocr

    directory = os.environ.get("EASYOCR_MODEL_DIR")
    if not directory:
        raise SystemExit("EASYOCR_MODEL_DIR is not set")
    Path(directory).mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        ["ko", "en"],
        gpu=False,
        model_storage_directory=directory,
        download_enabled=True,
    )
    reader.readtext(str(sample))
    saved = sorted(path.name for path in Path(directory).iterdir())
    # ko+en은 craft 검출기와 korean_g2 하나로 받는다. 영어 전용 pth는 따로 없다.
    if sum(name.endswith(".pth") for name in saved) < 2:
        raise SystemExit(f"EasyOCR 저장 결과 {directory}: {saved}")


def warm_paddle(sample: Path) -> None:
    from paddleocr import PaddleOCR
    from paddlex.inference.utils.official_models import official_models

    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0,
        device="cpu",
    )
    engine.predict(str(sample))
    for name in (
        "PP-OCRv6_small_det",
        "korean_PP-OCRv5_mobile_rec",
        "PP-LCNet_x1_0_textline_ori",
    ):
        official_models[name]


def verify() -> None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    layout = [
        path
        for suffix in ("*.safetensors", "*.bin", "*.pt")
        for path in hf_home.rglob(suffix)
    ]
    ocr = list((Path(rapidocr.__file__).parent / "models").glob(f"{OCR_LANGUAGE}*"))

    if not layout:
        sys.exit(f"docling 모델이 {hf_home}에 없다")
    if not ocr:
        sys.exit(f"{OCR_LANGUAGE} OCR 모델을 받지 못했다")

    easy_dir = Path(os.environ.get("EASYOCR_MODEL_DIR", ""))
    easy = list(easy_dir.glob("*.pth")) if easy_dir else []
    paddle_root = Path(
        os.environ.get(
            "PADDLE_PDX_CACHE_HOME",
            Path.home() / ".paddlex",
        )
    )
    paddle_models = [
        path
        for path in (paddle_root / "official_models").glob("*")
        if path.is_dir()
    ]
    if len(easy) < 2:
        names = sorted(path.name for path in easy_dir.iterdir()) if easy_dir.is_dir() else []
        sys.exit(f"EasyOCR 모델이 {easy_dir}에 없다: {names}")
    if len(paddle_models) < 3:
        sys.exit(f"Paddle 모델이 {paddle_root}에 {len(paddle_models)}개뿐이다")

    size = sum(path.stat().st_size for path in layout + ocr + easy) / 1024**2
    print(
        f"prefetched {len(layout)} docling + {len(ocr)} rapidocr"
        f" + {len(easy)} easyocr + {len(paddle_models)} paddle ({size:.0f} MiB)"
    )


if __name__ == "__main__":
    with TemporaryDirectory() as workdir:
        sample = Path(workdir) / "warmup.png"
        image = Image.new("RGB", (480, 120), "white")
        ImageDraw.Draw(image).text((20, 40), "관리규약 warmup", fill="black")
        image.save(sample)
        # parse_document가 아니라 컨버터를 직접 부른다. 글자를 못 읽어도 상관없고,
        # 모델을 못 받은 경우만 예외로 올라와야 한다.
        _converter().convert(sample)
        warm_easyocr(sample)
        warm_paddle(sample)
    verify()
