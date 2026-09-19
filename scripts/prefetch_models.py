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


def warm_up() -> None:
    with TemporaryDirectory() as workdir:
        sample = Path(workdir) / "warmup.png"
        image = Image.new("RGB", (480, 120), "white")
        ImageDraw.Draw(image).text((20, 40), "관리규약 warmup", fill="black")
        image.save(sample)
        # parse_document가 아니라 컨버터를 직접 부른다. 글자를 못 읽어도 상관없고,
        # 모델을 못 받은 경우만 예외로 올라와야 한다.
        _converter().convert(sample)


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

    size = sum(path.stat().st_size for path in layout + ocr) / 1024**2
    print(f"prefetched {len(layout)} docling + {len(ocr)} OCR files ({size:.0f} MiB)")


if __name__ == "__main__":
    warm_up()
    verify()
