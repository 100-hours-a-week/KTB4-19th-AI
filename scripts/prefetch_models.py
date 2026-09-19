"""빌드 단계에서 모델을 내려받아 이미지에 넣는다. 런타임에는 바깥으로 나가지 않는다.

내려받는 것은 docling 레이아웃·표 모델과 RapidOCR 한국어 모델이다. 목록을 따로 적는
대신 실제 파싱 경로를 한 번 태운다 — 설정이 바뀌면 받는 모델도 같이 따라온다.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from zipsai.errors import PdfParseError
from zipsai.indexing.parse import parse_document


def main() -> None:
    with TemporaryDirectory() as workdir:
        sample = Path(workdir) / "warmup.png"
        image = Image.new("RGB", (480, 120), "white")
        ImageDraw.Draw(image).text((20, 40), "관리규약 warmup", fill="black")
        image.save(sample)

        try:
            parse_document(sample)
        except PdfParseError:
            # 글자를 못 읽어도 된다. 목적은 모델을 캐시에 넣는 것뿐이다.
            pass

    print("prefetched docling and OCR models")


if __name__ == "__main__":
    main()
