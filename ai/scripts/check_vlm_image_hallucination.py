"""이미지 없는 턴에서 사진 관찰이 생성되지 않는지 확인한다.

사용: ai/에서 uv run python scripts/check_vlm_image_hallucination.py
"""

from zipsai.integrations.vlm import analyze_images

if __name__ == "__main__":
    analysis = analyze_images([], system_prompt="", user_prompt="")
    print(f"이미지 0장 → 관찰 {len(analysis.images)}건")
