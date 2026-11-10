"""AI 관상가 - Streamlit 메인 앱."""

import base64
import hashlib
import io
import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

from prompts import MOCK_RESULT, NO_FACE_MARKER, SYSTEM_PROMPT, USER_PROMPT

DEFAULT_MODEL = "gpt-4o-mini"
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_SIDE = 1024  # 전송 전 리사이즈할 긴 변 최대 길이(px)
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
DISCLAIMER = "※ 본 서비스는 재미로 보는 콘텐츠이며 과학적 근거가 없습니다."


class ImageProcessingError(Exception):
    """이미지를 읽거나 변환할 수 없을 때 발생하는 예외."""


# ─────────────────────────────── 설정 ───────────────────────────────


def get_config_value(name: str) -> str | None:
    """설정값을 Streamlit Secrets → .env(환경변수) 순서로 조회한다."""
    # 1) Streamlit Cloud: st.secrets (secrets.toml이 없으면 예외가 발생하므로 보호)
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass

    # 2) 로컬: .env 파일 로드 후 환경변수 조회 (이미 설정된 환경변수는 덮어쓰지 않음)
    load_dotenv(override=False)
    value = os.getenv(name, "").strip()
    return value or None


def is_mock_mode() -> bool:
    """MOCK_MODE가 켜져 있으면 API를 호출하지 않는다 (테스트용)."""
    return (get_config_value("MOCK_MODE") or "").lower() in ("1", "true", "yes", "on")


def get_api_key() -> str:
    """OpenAI API 키를 반환한다. 없으면 안내 후 앱 실행을 중단한다."""
    api_key = get_config_value("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다.\n\n"
            "- 로컬: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY`를 입력하세요.\n"
            "- Streamlit Cloud: App settings → Secrets에 `OPENAI_API_KEY`를 입력하세요."
        )
        st.stop()
    return api_key


def get_model() -> str:
    """사용할 OpenAI 모델명을 반환한다. 없으면 기본값을 사용한다."""
    return get_config_value("OPENAI_MODEL") or DEFAULT_MODEL


# ─────────────────────────── 이미지 처리 ───────────────────────────


def preprocess_image(image_bytes: bytes) -> bytes:
    """이미지를 회전 보정·RGB 변환·리사이즈 후 JPEG bytes로 반환한다."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()  # 손상 파일은 여기서 예외 발생
            # 첫 프레임만 사용 (애니메이션 WEBP 등)
            img.seek(0)
            img = ImageOps.exif_transpose(img)
            img = _to_rgb(img)
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageProcessingError("이미지를 읽을 수 없습니다.") from exc

    return buffer.getvalue()


def to_data_url(jpeg_bytes: bytes) -> str:
    """JPEG bytes를 OpenAI에 보낼 base64 data URL로 변환한다."""
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_processed_image(image_bytes: bytes, image_hash: str) -> bytes | None:
    """전처리 결과를 세션에 보관해 재실행마다 다시 처리하지 않는다. 실패 시 None."""
    cached = st.session_state.get("processed")
    if cached and cached[0] == image_hash:
        return cached[1]
    try:
        jpeg_bytes = preprocess_image(image_bytes)
    except ImageProcessingError:
        jpeg_bytes = None
    st.session_state.processed = (image_hash, jpeg_bytes)
    return jpeg_bytes


def _to_rgb(img: Image.Image) -> Image.Image:
    """투명 배경은 흰색으로 채우고, 모든 색상 모드를 RGB로 변환한다."""
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


# ─────────────────────────── 관상 분석 ───────────────────────────


def analyze_face(image_data_url: str) -> str:
    """OpenAI 비전 모델에 이미지를 보내 관상 풀이(마크다운)를 받는다."""
    if is_mock_mode():
        time.sleep(1)  # 로딩 표시 확인용
        return MOCK_RESULT

    client = openai.OpenAI(api_key=get_api_key(), timeout=60, max_retries=1)
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        max_completion_tokens=1500,
        temperature=0.8,
    )

    message = response.choices[0].message
    content = (message.content or "").strip()
    if not content:
        # 모델이 응답을 거부했거나 빈 응답을 준 경우
        raise ValueError("empty_response")
    return content


def describe_api_error(exc: Exception) -> str:
    """API 예외를 사용자용 한국어 안내 문구로 바꾼다. (키 등 민감 정보는 노출하지 않음)"""
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. `.env` 또는 Secrets의 `OPENAI_API_KEY`를 확인해 주세요."
    if isinstance(exc, openai.PermissionDeniedError):
        return "이 API 키로는 해당 모델을 사용할 권한이 없습니다. 모델명이나 계정 권한을 확인해 주세요."
    if isinstance(exc, openai.NotFoundError):
        return "설정한 모델을 찾을 수 없습니다. `OPENAI_MODEL` 값을 확인해 주세요."
    if isinstance(exc, openai.RateLimitError):
        return "요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도하거나 결제/한도 설정을 확인해 주세요."
    if isinstance(exc, openai.APITimeoutError):
        return "응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인해 주세요."
    if isinstance(exc, openai.BadRequestError):
        return "요청을 처리할 수 없습니다. 다른 사진으로 다시 시도해 주세요."
    if isinstance(exc, ValueError) and str(exc) == "empty_response":
        return "관상가가 이 사진에 대한 풀이를 하지 못했습니다. 얼굴이 잘 보이는 다른 사진으로 시도해 주세요."
    return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ──────────────────────────────── UI ────────────────────────────────

# 최소한의 사용자 정의 CSS: 직접 만든 HTML 클래스(.gs-*) 위주로 적용해
# Streamlit 내부 클래스명 변경에 덜 민감하게 한다. 글자색은 테마를 그대로 상속한다.
CUSTOM_CSS = """
<style>
.gs-hero { text-align: center; padding: 1.2rem 0 0.4rem; }
.gs-hero .gs-seal {
    display: inline-flex; align-items: center; justify-content: center;
    width: 3.4rem; height: 3.4rem; border-radius: 0.6rem;
    border: 2px solid rgba(196, 85, 58, 0.85); color: rgb(196, 85, 58);
    font-size: 1.5rem; font-weight: 700; transform: rotate(-4deg);
    margin-bottom: 0.6rem;
}
.gs-hero h1 { font-size: 2.3rem; margin: 0; padding: 0; letter-spacing: 0.08em; }
.gs-hero p { opacity: 0.8; margin: 0.4rem 0 0; }
.gs-ornament {
    height: 1px; margin: 1.1rem auto 0.4rem; width: 60%;
    background: linear-gradient(90deg, transparent, rgba(212, 168, 75, 0.9), transparent);
}
.gs-tip { text-align: center; padding: 0.2rem 0; }
.gs-tip .gs-tip-icon { font-size: 1.6rem; }
.gs-tip .gs-tip-title { font-weight: 700; margin-top: 0.2rem; }
.gs-tip .gs-tip-desc { font-size: 0.85rem; opacity: 0.75; }
.gs-label { font-size: 0.8rem; letter-spacing: 0.15em; opacity: 0.7; margin-bottom: 0.2rem; }
.gs-footer { text-align: center; font-size: 0.8rem; opacity: 0.65; line-height: 1.7; padding-bottom: 1rem; }
</style>
"""

# 결과 마크다운의 섹션 제목 키워드
SECTION_SUMMARY = "총평"
SECTION_PARTS = "부위별"
SECTION_FORTUNE = "운세"
SECTION_ADVICE = "한마디"

INPUT_UPLOAD = "📁 사진 업로드"
INPUT_CAMERA = "📷 카메라 촬영"


def render_header() -> None:
    """앱 상단 헤더 (낙관 도장 + 제목 + 소개)."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="gs-hero">'
        '<div class="gs-seal">相</div>'
        "<h1>AI 관상가</h1>"
        "<p>얼굴 사진 한 장으로 보는 전통 관상 풀이</p>"
        '<div class="gs-ornament"></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_tips() -> None:
    """좋은 사진을 고르는 3가지 팁."""
    tips = [
        ("🧑", "정면", "고개를 돌리지 않은 정면 얼굴"),
        ("💡", "밝은 조명", "그림자 없이 밝고 선명하게"),
        ("🖼️", "얼굴 전체", "이마부터 턱·귀까지 보이게"),
    ]
    for col, (icon, title, desc) in zip(st.columns(3), tips):
        with col, st.container(border=True):
            st.markdown(
                f'<div class="gs-tip"><div class="gs-tip-icon">{icon}</div>'
                f'<div class="gs-tip-title">{title}</div>'
                f'<div class="gs-tip-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )


def get_input_image() -> bytes | None:
    """업로드 또는 카메라 촬영으로 이미지를 받아 bytes로 반환한다."""
    # 위젯 key에 버전을 붙여 「처음부터 다시」 시 입력 위젯을 비울 수 있게 함
    version = st.session_state.get("input_version", 0)

    # 업로드/촬영 위젯이 동시에 값을 가지면 어느 쪽을 쓸지 모호하므로 방식을 하나 고르게 함
    method = st.segmented_control(
        "사진 입력 방식",
        [INPUT_UPLOAD, INPUT_CAMERA],
        default=INPUT_UPLOAD,
        key=f"method_{version}",
        width="stretch",
    )

    if method == INPUT_CAMERA:
        file = st.camera_input("카메라로 얼굴을 촬영하세요", key=f"camera_{version}")
    else:
        # 선택을 해제(None)해도 업로드 방식으로 취급
        file = st.file_uploader(
            f"얼굴이 잘 보이는 정면 사진을 올려주세요 (JPG · PNG · WEBP, 최대 {MAX_UPLOAD_MB}MB)",
            type=ALLOWED_TYPES,
            key=f"uploader_{version}",
            max_upload_size=MAX_UPLOAD_MB,  # 업로더 안내 문구도 10MB로 일치시킴
        )
    st.caption("📌 본인 또는 동의받은 사진만 사용하세요. 업로드한 이미지는 저장되지 않습니다.")

    if file is None:
        return None

    if file.size > MAX_UPLOAD_BYTES:
        st.warning(f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하의 사진을 올려주세요.")
        return None

    return file.getvalue()


def reset_result_if_image_changed(image_hash: str | None) -> None:
    """새 이미지가 들어오면 이전 분석 결과를 지운다."""
    if st.session_state.get("image_hash") != image_hash:
        st.session_state.image_hash = image_hash
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.pending_hash = None


def request_analysis() -> None:
    """버튼 콜백: 분석 대기 상태로 바꿔 버튼을 즉시 비활성화한다 (중복 클릭 방지)."""
    st.session_state.pending_hash = st.session_state.get("image_hash")


def reset_all() -> None:
    """「처음부터 다시」 콜백: 입력 위젯과 결과를 모두 초기화한다."""
    st.session_state.input_version = st.session_state.get("input_version", 0) + 1
    for key in ("image_hash", "result", "error", "pending_hash", "processed"):
        st.session_state.pop(key, None)


def run_analysis(jpeg_bytes: bytes) -> None:
    """전처리된 이미지로 관상 분석을 실행해 결과/오류를 session_state에 저장한다."""
    st.session_state.result = None
    st.session_state.error = None
    try:
        with st.spinner("관상가가 얼굴을 살펴보는 중입니다... 🔍"):
            st.session_state.result = analyze_face(to_data_url(jpeg_bytes))
    except Exception as exc:  # noqa: BLE001 - 모든 API 오류를 사용자 안내로 변환
        st.session_state.error = describe_api_error(exc)
    finally:
        st.session_state.pending_hash = None
    # 버튼을 다시 활성화된 상태로 그리기 위해 재실행 (결과/오류는 session_state에 보존)
    st.rerun()


def split_sections(markdown: str) -> dict[str, str]:
    """'## 제목' 기준으로 결과 마크다운을 {제목: 본문}으로 나눈다."""
    sections: dict[str, str] = {}
    title = None
    lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if title is not None:
                sections[title] = "\n".join(lines).strip()
            title, lines = line[3:].strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None:
        sections[title] = "\n".join(lines).strip()
    return sections


def find_section(sections: dict[str, str], keyword: str) -> str | None:
    """제목에 keyword가 들어간 섹션 본문을 찾는다."""
    for title, body in sections.items():
        if keyword in title:
            return body
    return None


def show_result(result: str, jpeg_bytes: bytes | None) -> None:
    """분석 결과를 카드·탭으로 표시한다. 형식이 다르면 원문 마크다운으로 보여준다."""
    if result.startswith(NO_FACE_MARKER):
        st.warning(result.removeprefix(NO_FACE_MARKER).strip(), icon="🙈")
        return

    sections = split_sections(result)
    summary = find_section(sections, SECTION_SUMMARY)
    parts = find_section(sections, SECTION_PARTS)
    fortune = find_section(sections, SECTION_FORTUNE)
    advice = find_section(sections, SECTION_ADVICE)

    # 예상한 형식이 아니면 원문 그대로 출력 (내용이 사라지지 않도록)
    if not (summary and parts and fortune):
        with st.container(border=True):
            st.markdown(result)
        return

    # 사진 + 총평 (넓은 화면은 2열, 좁은 화면에서는 Streamlit이 자동으로 1열 배치)
    photo_col, summary_col = st.columns([2, 3], vertical_alignment="center")
    with photo_col:
        if jpeg_bytes:
            # 모바일에서 1열로 쌓일 때 사진이 화면 전체로 커지지 않도록 고정 폭 사용
            st.image(jpeg_bytes, width=260)
    with summary_col, st.container(border=True):
        st.markdown('<div class="gs-label">관상 총평</div>', unsafe_allow_html=True)
        st.markdown(f"##### 🔮 {summary}")

    parts_tab, fortune_tab = st.tabs(["👤 부위별 풀이", "🍀 운세"])
    with parts_tab:
        st.markdown(parts)
    with fortune_tab:
        st.markdown(fortune)

    if advice:
        st.success(f"**관상가의 한마디**  \n{advice}", icon="💬")

    # 예상하지 못한 추가 섹션이 있으면 놓치지 않도록 펼침 영역에 표시
    known = (SECTION_SUMMARY, SECTION_PARTS, SECTION_FORTUNE, SECTION_ADVICE)
    extra = {t: b for t, b in sections.items() if not any(k in t for k in known)}
    if extra:
        with st.expander("더 보기"):
            for title, body in extra.items():
                st.markdown(f"**{title}**\n\n{body}")


def render_footer() -> None:
    """면책 고지와 개인정보 안내."""
    st.markdown(
        '<div class="gs-ornament"></div>'
        f'<div class="gs-footer">{DISCLAIMER}<br>'
        "업로드한 사진은 분석에만 사용되며 서버에 저장되지 않습니다.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    render_header()

    # 설정 확인 (MOCK_MODE가 아니면 키가 없을 때 여기서 중단)
    if is_mock_mode():
        st.caption("🧪 MOCK_MODE: 실제 API를 호출하지 않습니다.")
    else:
        get_api_key()

    # 결과가 없을 때만 촬영 팁 표시 (결과 화면을 간결하게)
    if not st.session_state.get("result"):
        render_tips()

    image_bytes = get_input_image()
    image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    reset_result_if_image_changed(image_hash)
    has_result = bool(st.session_state.get("result"))

    # 업로드 직후 한 번 전처리: 손상 파일을 미리 걸러내고, 회전 보정된 사진으로 미리보기
    jpeg_bytes = get_processed_image(image_bytes, image_hash) if image_bytes else None
    if image_bytes and jpeg_bytes is None:
        st.error("이미지를 읽을 수 없습니다. 손상되지 않은 JPG · PNG · WEBP 사진을 올려주세요.")
    if jpeg_bytes and not has_result:
        # 결과가 나오면 사진은 결과 영역에 함께 보여주므로 미리보기는 생략
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(jpeg_bytes, caption="선택한 사진", width="stretch")

    is_pending = image_hash is not None and st.session_state.get("pending_hash") == image_hash
    st.button(
        "🔮 관상 보기",
        type="primary",
        disabled=jpeg_bytes is None or is_pending,
        on_click=request_analysis,
        width="stretch",
    )

    if is_pending and jpeg_bytes:
        run_analysis(jpeg_bytes)

    if st.session_state.get("error"):
        st.error(st.session_state.error)

    if has_result:
        st.markdown('<div class="gs-ornament"></div>', unsafe_allow_html=True)
        show_result(st.session_state.result, jpeg_bytes)
        st.button("🔄 처음부터 다시", on_click=reset_all, width="stretch")

    render_footer()


if __name__ == "__main__":
    main()
