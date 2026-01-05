import asyncio
import cgi
import os
import shutil
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.i18n import I18nManager, get_i18n_manager, set_language
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    X302AITranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

logger = logging.getLogger(__name__)

BABELDOC_MODEL = OnnxModel.load_available()

# Initialize i18n manager
# Get saved language preference from config, default to Traditional Chinese
saved_language = ConfigManager.get_language("繁體中文 (Traditional Chinese)")
i18n_manager = get_i18n_manager(saved_language)
set_language(saved_language)


def t(key: str, default: str = "") -> str:
    """Shortcut function for translation"""
    return i18n_manager.translate(key, default)


# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# i18n display keys for language names (values remain stable internal tokens)
lang_ui_key_map = {
    "English": "lang_english",
    "Simplified Chinese": "lang_simplified_chinese",
    "Traditional Chinese": "lang_traditional_chinese",
    "French": "lang_french",
    "German": "lang_german",
    "Japanese": "lang_japanese",
    "Korean": "lang_korean",
    "Russian": "lang_russian",
    "Spanish": "lang_spanish",
    "Italian": "lang_italian",
}


def get_lang_choices() -> list[tuple[str, str]]:
    return [(t(lang_ui_key_map[name]), name) for name in lang_map.keys()]

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["Google", "Bing"]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    enabled_services = default_services + enabled_services
else:
    enabled_services = list(service_map.keys())


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    translate_table_text,
    no_watermark,
    vfont,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error(t("error_recaptcha_fail"))

    progress(0, desc=t("progress_starting"))

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error(t("error_no_input"))
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error(t("error_no_input"))
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys
    # Persist latest env values for this translator so they survive reloads
    ConfigManager.set_translator_by_name(translator.name, _envs)

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        progress(t.n / t.total, desc=desc)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "translate_table_text": translate_table_text,
        "no_watermark": no_watermark,
        "vfont": vfont,
        "model": ModelInstance.value,
    }

    try:
        if use_babeldoc:
            return babeldoc_translate_file(**param)
        translate(**param)
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")
    print(f"Files after translation: {os.listdir(output)}")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig, WatermarkOutputMode
    
    # Import RapidOCRModel for table text translation
    table_model = None
    if kwargs.get("translate_table_text", False):
        try:
            from babeldoc.docvision.table_detection.rapidocr import RapidOCRModel
            table_model = RapidOCRModel()
            logger.info("Table text translation enabled with RapidOCR")
        except ImportError as e:
            logger.warning(f"RapidOCR not installed. Install with: pip install rapidocr_onnxruntime")
            logger.warning(f"Table text translation disabled. Error: {e}")
        except Exception as e:
            logger.warning(f"Failed to load RapidOCRModel: {e}")
            logger.warning("Table text translation disabled due to initialization error")

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
        X302AITranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        # Determine watermark output mode
        watermark_mode = WatermarkOutputMode.NoWatermark if kwargs.get("no_watermark", True) else WatermarkOutputMode.Watermarked

        # Check if translator is LLM-based (supports rich text translation)
        is_llm_translator = isinstance(translator, (OpenAITranslator, AzureOpenAITranslator, OllamaTranslator, GeminiTranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, DeepseekTranslator, GroqTranslator, GrokTranslator, OpenAIlikedTranslator, QwenMtTranslator, X302AITranslator))

        # ============================================================
        # BabelDOC Configuration - Optimized based on community best practices
        # References:
        # - https://github.com/funstory-ai/BabelDOC
        # - https://funstory-ai.github.io/BabelDOC/
        # - GitHub Issues #89, #254, #369
        # ============================================================
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,

            # ============ Rich Text Translation ============
            # For LLM translators (OpenAI, Azure, DeepSeek, etc.):
            #   - Keep rich text enabled for better translation quality
            # For traditional translators (Google, Bing, DeepL):
            #   - Disable rich text for compatibility
            disable_rich_text_translate=not is_llm_translator,

            # ============ Font & Cleanup ============
            skip_clean=kwargs["skip_subset_fonts"],

            # ============ Progress Reporting ============
            report_interval=0.5,

            # ============ Table Processing ============
            table_model=table_model,

            # ============ Watermark ============
            watermark_output_mode=watermark_mode,

            # ============ Paragraph Detection Optimization ============
            # min_text_length: Set to 1 to ensure all text blocks are translated
            # (Default is 5, which may skip short titles or labels)
            min_text_length=1,

            # split_short_lines: MUST be False to prevent word splitting
            # When True, may cause "What practices" to split into "What prac" + "tices"
            split_short_lines=False,

            # short_line_split_factor: Controls paragraph splitting sensitivity
            # Lower value = less aggressive splitting = fewer split words
            # Default: 0.8, Optimized: 0.5 to reduce false paragraph breaks
            short_line_split_factor=0.5,

            # ============ Translation Quality ============
            # dual_translate_first: Translate dual (bilingual) version first
            # This improves translation quality for the mono version
            dual_translate_first=True,

            # ============ Compatibility ============
            # Note: enhance_compatibility enables skip_clean, dual_translate_first,
            # AND disable_rich_text_translate. We set individual options instead
            # to maintain rich text for LLM translators.
            # enhance_compatibility=False (not set, using individual options)
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # Initialize variables to track translation result
            file_mono = None
            file_dual = None
            translate_error = None

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "translate_error":
                        # Capture translation error
                        translate_error = event.get("message", "Unknown translation error")
                        logger.error(f"Translation error: {translate_error}")
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()

            # Check if translation was successful
            if file_mono is None or file_dual is None:
                error_msg = translate_error or t("error_translation_failed")
                logger.error(f"Translation failed: {error_msg}")
                raise gr.Error(f"{t('error_translation_failed')}: {error_msg}")

            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    /* Hide footer and set base background */
    footer {visibility: hidden}
    body, .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #f6f7fb;
        min-height: 100vh;
    }

    .gradio-container {
        max-width: 1600px;
        width: 100%;
        margin: 1rem auto;
        padding: 0 1.5rem 2.5rem;
    }
    @media (min-width: 1400px) {
        .gradio-container { max-width: 90vw; }
    }
    .gradio-container .prose { max-width: none; }

    /* Subtle text */
    .secondary-text { color: #6b7280 !important; font-size: 0.9rem; }
    .env-warning { color: #d97706 !important; font-weight: 600; }
    .env-success { color: #16a34a !important; font-weight: 600; }

    /* Header */
    .app-header { text-align: center; padding: 1rem 0 0.5rem; }
    .app-header h1, .app-header h2 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.25rem 0 0.5rem;
    }

    /* Top controls */
    .top-controls { margin: 0 0 0.75rem 0; }

    /* Cards */
    .panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.25rem 1.1rem;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    }
    .panel .prose h2 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.75rem 0;
    }
    .panel .prose h4 {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin: 0.75rem 0 0.5rem 0;
    }
    .panel-left { border-left: none; }
    .panel-right { border-left: none; }

    /* Layout */
    .main-row { gap: 1.25rem; align-items: stretch !important; }

    /* Inputs */
    .input-file {
        border: 1.6px dashed #1f7aec !important;
        border-radius: 10px !important;
        background: #f8fbff !important;
        padding: 1.25rem !important;
    }
    .input-file:hover { border-color: #0e42d2 !important; background: #f0f5ff !important; }

    .gradio-textbox, .gradio-dropdown, .gradio-radio {
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
        transition: box-shadow 0.15s ease, border-color 0.15s ease !important;
    }
    .gradio-textbox:focus, .gradio-dropdown:focus, .gradio-radio:focus {
        border-color: #1f7aec !important;
        box-shadow: 0 0 0 3px rgba(31, 122, 236, 0.15) !important;
    }
    .gradio-checkbox input[type="checkbox"] { border-radius: 4px !important; cursor: pointer; }

    /* Buttons */
    .gradio-button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.2rem !important;
        letter-spacing: 0.01em;
        border: none !important;
        transition: transform 0.12s ease, box-shadow 0.15s ease !important;
    }
    .gradio-button:not(.secondary) {
        background: linear-gradient(135deg, #1f7aec 0%, #0e42d2 100%) !important;
        color: #ffffff !important;
    }
    .gradio-button:not(.secondary):hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 16px rgba(31, 122, 236, 0.25);
    }
    .gradio-button.secondary {
        background: #f3f4f6 !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
    }
    .gradio-button.secondary:hover {
        background: #e5e7eb !important;
    }
    .action-row { gap: 0.75rem; display: flex; }
    .action-row button { flex: 1; min-height: 44px; }

    /* Progress */
    .progress-bar-wrap, .progress-bar {
        border-radius: 10px !important;
        overflow: hidden;
    }
    .progress-bar { background: linear-gradient(90deg, #1f7aec 0%, #0e42d2 100%) !important; }

    /* Accordion */
    .gradio-accordion { border-radius: 10px !important; border: 1px solid #e5e7eb !important; overflow: hidden; }
    .gradio-accordion-header {
        background: #f8fafc !important;
        border-bottom: 1px solid #e5e7eb !important;
        padding: 0.9rem 1rem !important;
        font-weight: 650 !important;
        color: #0f172a !important;
    }

    /* Preview */
    .pdf-canvas canvas { width: 100%; border-radius: 8px; }
    .gradio-file { border-radius: 10px !important; background: #f9fafb !important; border: 1px solid #e5e7eb !important; }

    /* Responsive */
    @media (max-width: 1024px) {
        .main-row { flex-direction: column; }
        .panel { padding: 1rem; }
        .app-header h1, .app-header h2 { font-size: 1.6rem; }
    }
    @media (max-width: 640px) {
        .gradio-container { margin: 1rem auto; padding: 0 0.75rem 1.25rem; }
        .panel { padding: 0.85rem; border-radius: 10px; }
        .action-row button { font-size: 0.9rem; padding: 0.55rem 1rem !important; }
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>{t("tech_details")}</summary>
                    - {t("tech_details_github")}<br>
                    - {t("tech_details_babeldoc")}<br>
                    - {t("tech_details_gui")}<br>
                    - {t("tech_details_version")} {__version__} <br>
                    - {t("tech_details_babeldoc_version")} {babeldoc_version}
                """
cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title=t("app_title"),
    theme=gr.themes.Soft(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    header_md = gr.Markdown(t("header_github"), elem_classes=["app-header"])

    # Top language selector for quick access
    with gr.Row(elem_classes=["top-controls"]):
        language_selector = gr.Dropdown(
            label=t("language_label"),
            choices=i18n_manager.get_all_languages(),
            value=i18n_manager.get_current_language(),
            scale=1,
        )

    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(scale=1, elem_classes=["panel", "panel-left"]):
            file_section_title = gr.Markdown(
                t("file_section") if flag_demo else t("file_section_no_limit")
            )
            file_type = gr.Radio(
                # Display is translated, value remains stable for callbacks ("File"/"Link")
                choices=[(t("type_file"), "File"), (t("type_link"), "Link")],
                label=t("type_label"),
                value="File",
            )
            file_input = gr.File(
                label=t("file_label"),
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label=t("link_label"),
                visible=False,
                interactive=True,
            )
            option_section_title = gr.Markdown("## " + t("option_section"))
            service = gr.Dropdown(
                label=t("service_label"),
                choices=enabled_services,
                value=enabled_services[0],
            )
            envs = []
            for i in range(4):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label=t("translate_from_label"),
                    choices=get_lang_choices(),
                    value=ConfigManager.get("PDF2ZH_LANG_FROM", "English"),
                )
                lang_to = gr.Dropdown(
                    label=t("translate_to_label"),
                    choices=get_lang_choices(),
                    value=ConfigManager.get("PDF2ZH_LANG_TO", "Traditional Chinese"),
                )
            page_range = gr.Radio(
                # Display is translated, value remains stable for business logic
                choices=[
                    (t("page_all"), "All"),
                    (t("page_first"), "First"),
                    (t("page_first_5"), "First 5 pages"),
                    (t("page_others"), "Others"),
                ],
                label=t("pages_label"),
                value="All",
            )

            page_input = gr.Textbox(
                label=t("page_range_label"),
                visible=False,
                interactive=True,
            )

            with gr.Accordion(t("more_options"), open=False) as more_options_acc:
                experimental_title = gr.Markdown("#### " + t("experimental_label"))
                threads = gr.Textbox(
                    label=t("threads_label"), interactive=True, value="6"
                )
                skip_subset_fonts = gr.Checkbox(
                    label=t("skip_fonts_label"), interactive=True, value=True
                )
                ignore_cache = gr.Checkbox(
                    label=t("ignore_cache_label"), interactive=True, value=False
                )
                translate_table_text = gr.Checkbox(
                    label=t("translate_table_label"), interactive=True, value=False
                )
                no_watermark = gr.Checkbox(
                    label=t("no_watermark_label"), interactive=True, value=True
                )
                vfont = gr.Textbox(
                    label=t("vfont_label"),
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label=t("custom_prompt_label"), interactive=True, visible=False
                )
                use_babeldoc = gr.Checkbox(
                    label=t("use_babeldoc_label"), interactive=True, value=True
                )
                envs.append(prompt)

            # Track current service for env saving
            current_service_state = gr.State(value=enabled_services[0])

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                _envs = []
                for i in range(5):
                    _envs.append(gr.update(visible=False, value=""))
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                # Return envs + updated service state
                return _envs + [service]

            def save_env_on_change(service_name, env_idx, value):
                """Save env value to config when user changes it"""
                if not service_name or service_name not in service_map:
                    return value
                translator = service_map[service_name]
                env_keys = list(translator.envs.keys())
                if env_idx < len(env_keys):
                    key = env_keys[env_idx]
                    # Don't save masked values
                    if value != "***":
                        saved_envs = ConfigManager.get_translator_by_name(translator.name) or {}
                        saved_envs[key] = value
                        ConfigManager.set_translator_by_name(translator.name, saved_envs)
                return value

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            def on_language_change(language):
                """Save language preference and update UI"""
                set_language(language)
                ConfigManager.set_language(language)

                # Recompute tech details string with new language
                new_tech_details = f"""
                    <summary>{t("tech_details")}</summary>
                    - {t("tech_details_github")}<br>
                    - {t("tech_details_babeldoc")}<br>
                    - {t("tech_details_gui")}<br>
                    - {t("tech_details_version")} {__version__} <br>
                    - {t("tech_details_babeldoc_version")} {babeldoc_version}
                """

                # Return updates for key UI components
                return [
                    gr.update(value=t("header_github")),  # header_md
                    gr.update(value=t("file_section") if flag_demo else t("file_section_no_limit")),  # file_section_title
                    gr.update(
                        label=t("type_label"),
                        choices=[(t("type_file"), "File"), (t("type_link"), "Link")],
                    ),  # file_type
                    gr.update(value="## " + t("option_section")),  # option_section_title
                    gr.update(value="#### " + t("experimental_label")),  # experimental_title
                    gr.update(label=t("service_label")),  # service
                    gr.update(label=t("translate_from_label"), choices=get_lang_choices()),  # lang_from
                    gr.update(label=t("translate_to_label"), choices=get_lang_choices()),  # lang_to
                    gr.update(
                        label=t("pages_label"),
                        choices=[
                            (t("page_all"), "All"),
                            (t("page_first"), "First"),
                            (t("page_first_5"), "First 5 pages"),
                            (t("page_others"), "Others"),
                        ],
                    ),  # page_range
                    gr.update(label=t("page_range_label")),  # page_input
                    gr.update(label=t("more_options")),  # more_options_acc
                    gr.update(label=t("threads_label")),  # threads
                    gr.update(label=t("skip_fonts_label")),  # skip_subset_fonts
                    gr.update(label=t("ignore_cache_label")),  # ignore_cache
                    gr.update(label=t("vfont_label")),  # vfont
                    gr.update(label=t("custom_prompt_label")),  # prompt
                    gr.update(label=t("use_babeldoc_label")),  # use_babeldoc
                    gr.update(label=t("file_label")),  # file_input
                    gr.update(label=t("link_label")),  # link_input
                    gr.update(value=t("translated_section")),  # output_title
                    gr.update(label=t("download_mono_label")),  # output_file_mono
                    gr.update(label=t("download_dual_label")),  # output_file_dual
                    gr.update(label=t("recaptcha_label")),  # recaptcha_response
                    gr.update(label=t("language_label"), value=language),  # language_selector
                    gr.update(value=t("translate_btn")),  # translate_btn
                    gr.update(value=t("cancel_btn")),  # cancellation_btn
                    gr.update(value=new_tech_details),  # tech_details_tog
                    gr.update(value="## " + t("preview_section")),  # preview_section_title
                    gr.update(label=t("preview_section")),  # preview
                ]

            output_title = gr.Markdown(t("translated_section"), visible=False)
            output_file_mono = gr.File(
                label=t("download_mono_label"), visible=False
            )
            output_file_dual = gr.File(
                label=t("download_dual_label"), visible=False
            )
            recaptcha_response = gr.Textbox(
                label=t("recaptcha_label"), elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')

            with gr.Row(elem_classes=["action-row"]):
                translate_btn = gr.Button(t("translate_btn"), variant="primary", scale=2)
                cancellation_btn = gr.Button(t("cancel_btn"), variant="secondary", scale=1)
            tech_details_tog = gr.Markdown(
                tech_details_string,
                elem_classes=["secondary-text"],
            )
            page_range.select(on_select_page, page_range, page_input)
            service.select(
                on_select_service,
                service,
                envs + [current_service_state],
            )
            # Add change handlers for env inputs to save values immediately
            for idx, env_input in enumerate(envs[:4]):  # First 4 are the env textboxes
                env_input.change(
                    lambda svc, val, i=idx: save_env_on_change(svc, i, val),
                    inputs=[current_service_state, env_input],
                    outputs=None,
                )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2, elem_classes=["panel", "panel-right"]):
            preview_section_title = gr.Markdown("## " + t("preview_section"))
            preview = PDF(label=t("preview_section"), visible=True, height=2000)

    # Bind language change after all components are created (so outputs exist)
    language_selector.change(
        on_language_change,
        inputs=language_selector,
        outputs=[
            header_md,
            file_section_title,
            file_type,
            option_section_title,
            experimental_title,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            more_options_acc,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            prompt,
            use_babeldoc,
            file_input,
            link_input,
            output_title,
            output_file_mono,
            output_file_dual,
            recaptcha_response,
            language_selector,
            translate_btn,
            cancellation_btn,
            tech_details_tog,
            preview_section_title,
            preview,
        ],
    )

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    state = gr.State({"session_id": None})

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            translate_table_text,
            no_watermark,
            vfont,
            use_babeldoc,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
    ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)
    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
    else:
        if len(user_list) == 0:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True, inbrowser=True, share=True, server_port=server_port
                    )
        else:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True,
                        inbrowser=True,
                        share=True,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
