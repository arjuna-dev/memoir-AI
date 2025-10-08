"""
Context window sizes for various LLM models supported by Pydantic AI.
"""


class Model:
    def __init__(self, name: str, context_length: int, litellm_name: str = ""):
        self.name = name
        self.context_length = context_length
        self.litellm_name = litellm_name


class Models:
    # Anthropic models
    anthropic_claude_3_5_haiku_20241022 = Model(
        "anthropic:claude-3-5-haiku-20241022", 200_000, "claude-3-5"
    )
    anthropic_claude_3_5_haiku_latest = Model(
        "anthropic:claude-3-5-haiku-latest", 200_000, "claude-3-5"
    )
    anthropic_claude_3_5_sonnet_20240620 = Model(
        "anthropic:claude-3-5-sonnet-20240620", 200_000, "claude-3-5"
    )
    anthropic_claude_3_5_sonnet_20241022 = Model(
        "anthropic:claude-3-5-sonnet-20241022", 200_000, "claude-3-5"
    )
    anthropic_claude_3_5_sonnet_latest = Model(
        "anthropic:claude-3-5-sonnet-latest", 200_000, "claude-3-5"
    )
    anthropic_claude_3_7_sonnet_20250219 = Model(
        "anthropic:claude-3-7-sonnet-20250219", 200_000, "claude-3-7"
    )
    anthropic_claude_3_7_sonnet_latest = Model(
        "anthropic:claude-3-7-sonnet-latest", 200_000, "claude-3-7"
    )
    anthropic_claude_3_haiku_20240307 = Model(
        "anthropic:claude-3-haiku-20240307", 200_000, "claude-3"
    )
    anthropic_claude_3_opus_20240229 = Model(
        "anthropic:claude-3-opus-20240229", 200_000, "claude-3"
    )
    anthropic_claude_3_opus_latest = Model(
        "anthropic:claude-3-opus-latest", 200_000, "claude-3"
    )
    anthropic_claude_4_opus_20250514 = Model(
        "anthropic:claude-4-opus-20250514", 200_000, "claude-opus-4"
    )
    anthropic_claude_4_sonnet_20250514 = Model(
        "anthropic:claude-4-sonnet-20250514", 200_000, "claude-sonnet-4"
    )
    anthropic_claude_opus_4_0 = Model(
        "anthropic:claude-opus-4-0", 200_000, "claude-opus-4"
    )
    anthropic_claude_opus_4_1_20250805 = Model(
        "anthropic:claude-opus-4-1-20250805", 200_000, "claude-opus-4-1"
    )
    anthropic_claude_opus_4_20250514 = Model(
        "anthropic:claude-opus-4-20250514", 200_000, "claude-opus-4"
    )
    anthropic_claude_sonnet_4_0 = Model(
        "anthropic:claude-sonnet-4-0", 1_000_000, "claude-sonnet-4"
    )
    anthropic_claude_sonnet_4_20250514 = Model(
        "anthropic:claude-sonnet-4-20250514", 200_000, "claude-sonnet-4"
    )
    anthropic_claude_sonnet_4_5 = Model(
        "anthropic:claude-sonnet-4-5", 200_000, "claude-sonnet-4-5"
    )
    anthropic_claude_sonnet_4_5_20250929 = Model(
        "anthropic:claude-sonnet-4-5-20250929", 200_000, "claude-sonnet-4-5"
    )

    # Bedrock models
    bedrock_amazon_titan_tg1_large = Model(
        "bedrock:amazon.titan-tg1-large", 32_000
    )  # ?
    bedrock_amazon_titan_text_lite_v1 = Model(
        "bedrock:amazon.titan-text-lite-v1", 4_000
    )
    bedrock_amazon_titan_text_express_v1 = Model(
        "bedrock:amazon.titan-text-express-v1", 8_000
    )
    bedrock_us_amazon_nova_pro_v1_0 = Model("bedrock:us.amazon.nova-pro-v1:0", 300_000)
    bedrock_us_amazon_nova_lite_v1_0 = Model(
        "bedrock:us.amazon.nova-lite-v1:0", 300_000
    )
    bedrock_us_amazon_nova_micro_v1_0 = Model(
        "bedrock:us.amazon.nova-micro-v1:0", 128_000
    )
    bedrock_anthropic_claude_3_5_sonnet_20241022_v2_0 = Model(
        "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0", 200_000
    )
    bedrock_us_anthropic_claude_3_5_sonnet_20241022_v2_0 = Model(
        "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0", 200_000
    )
    bedrock_anthropic_claude_3_5_haiku_20241022_v1_0 = Model(
        "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_5_haiku_20241022_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0", 200_000
    )
    bedrock_anthropic_claude_instant_v1 = Model(
        "bedrock:anthropic.claude-instant-v1", 100_000
    )
    bedrock_anthropic_claude_v2_1 = Model("bedrock:anthropic.claude-v2:1", 100_000)
    bedrock_anthropic_claude_v2 = Model("bedrock:anthropic.claude-v2", 100_000)
    bedrock_anthropic_claude_3_sonnet_20240229_v1_0 = Model(
        "bedrock:anthropic.claude-3-sonnet-20240229-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_sonnet_20240229_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0", 200_000
    )
    bedrock_anthropic_claude_3_haiku_20240307_v1_0 = Model(
        "bedrock:anthropic.claude-3-haiku-20240307-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_haiku_20240307_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-haiku-20240307-v1:0", 200_000
    )
    bedrock_anthropic_claude_3_opus_20240229_v1_0 = Model(
        "bedrock:anthropic.claude-3-opus-20240229-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_opus_20240229_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-opus-20240229-v1:0", 200_000
    )
    bedrock_anthropic_claude_3_5_sonnet_20240620_v1_0 = Model(
        "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_5_sonnet_20240620_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0", 200_000
    )
    bedrock_anthropic_claude_3_7_sonnet_20250219_v1_0 = Model(
        "bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_3_7_sonnet_20250219_v1_0 = Model(
        "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0", 200_000
    )
    bedrock_anthropic_claude_opus_4_20250514_v1_0 = Model(
        "bedrock:anthropic.claude-opus-4-20250514-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_opus_4_20250514_v1_0 = Model(
        "bedrock:us.anthropic.claude-opus-4-20250514-v1:0", 200_000
    )
    bedrock_anthropic_claude_sonnet_4_20250514_v1_0 = Model(
        "bedrock:anthropic.claude-sonnet-4-20250514-v1:0", 200_000
    )
    bedrock_us_anthropic_claude_sonnet_4_20250514_v1_0 = Model(
        "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0", 200_000
    )
    bedrock_cohere_command_text_v14 = Model("bedrock:cohere.command-text-v14", 4_000)
    bedrock_cohere_command_r_v1_0 = Model("bedrock:cohere.command-r-v1:0", 128_000)
    bedrock_cohere_command_r_plus_v1_0 = Model(
        "bedrock:cohere.command-r-plus-v1:0", 128_000
    )
    bedrock_cohere_command_light_text_v14 = Model(
        "bedrock:cohere.command-light-text-v14", 4_000
    )
    bedrock_meta_llama3_8b_instruct_v1_0 = Model(
        "bedrock:meta.llama3-8b-instruct-v1:0", 8_000
    )
    bedrock_meta_llama3_70b_instruct_v1_0 = Model(
        "bedrock:meta.llama3-70b-instruct-v1:0", 8_000
    )
    bedrock_meta_llama3_1_8b_instruct_v1_0 = Model(
        "bedrock:meta.llama3-1-8b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_1_8b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-1-8b-instruct-v1:0", 128_000
    )
    bedrock_meta_llama3_1_70b_instruct_v1_0 = Model(
        "bedrock:meta.llama3-1-70b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_1_70b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-1-70b-instruct-v1:0", 128_000
    )
    bedrock_meta_llama3_1_405b_instruct_v1_0 = Model(
        "bedrock:meta.llama3-1-405b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_2_11b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-2-11b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_2_90b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-2-90b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_2_1b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-2-1b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_2_3b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-2-3b-instruct-v1:0", 128_000
    )
    bedrock_us_meta_llama3_3_70b_instruct_v1_0 = Model(
        "bedrock:us.meta.llama3-3-70b-instruct-v1:0", 128_000
    )
    bedrock_mistral_mistral_7b_instruct_v0_2 = Model(
        "bedrock:mistral.mistral-7b-instruct-v0:2", 32_000
    )
    bedrock_mistral_mixtral_8x7b_instruct_v0_1 = Model(
        "bedrock:mistral.mixtral-8x7b-instruct-v0:1", 32_000
    )
    bedrock_mistral_mistral_large_2402_v1_0 = Model(
        "bedrock:mistral.mistral-large-2402-v1:0", 32_000
    )
    bedrock_mistral_mistral_large_2407_v1_0 = Model(
        "bedrock:mistral.mistral-large-2407-v1:0", 128_000
    )

    # Cerebras models
    cerebras_gpt_oss_120b = Model("cerebras:gpt-oss-120b", 131_072)
    cerebras_llama3_1_8b = Model("cerebras:llama3.1-8b", 128_000)
    cerebras_llama_3_3_70b = Model("cerebras:llama-3.3-70b", 128_000)
    cerebras_llama_4_scout_17b_16e_instruct = Model(
        "cerebras:llama-4-scout-17b-16e-instruct", 10_000_000
    )
    cerebras_llama_4_maverick_17b_128e_instruct = Model(
        "cerebras:llama-4-maverick-17b-128e-instruct", 1_000_000
    )
    cerebras_qwen_3_235b_a22b_instruct_2507 = Model(
        "cerebras:qwen-3-235b-a22b-instruct-2507", 128_000
    )
    cerebras_qwen_3_32b = Model("cerebras:qwen-3-32b", 128_000)
    cerebras_qwen_3_coder_480b = Model("cerebras:qwen-3-coder-480b", 128_000)
    cerebras_qwen_3_235b_a22b_thinking_2507 = Model(
        "cerebras:qwen-3-235b-a22b-thinking-2507", 128_000
    )

    # Cohere models
    cohere_c4ai_aya_expanse_32b = Model("cohere:c4ai-aya-expanse-32b", 128_000)
    cohere_c4ai_aya_expanse_8b = Model("cohere:c4ai-aya-expanse-8b", 128_000)
    cohere_command = Model("cohere:command", 4_000)
    cohere_command_light = Model("cohere:command-light", 4_000)
    cohere_command_light_nightly = Model("cohere:command-light-nightly", 4_000)
    cohere_command_nightly = Model("cohere:command-nightly", 4_000)
    cohere_command_r = Model("cohere:command-r", 128_000)
    cohere_command_r_03_2024 = Model("cohere:command-r-03-2024", 128_000)
    cohere_command_r_08_2024 = Model("cohere:command-r-08-2024", 128_000)
    cohere_command_r_plus = Model("cohere:command-r-plus", 128_000)
    cohere_command_r_plus_04_2024 = Model("cohere:command-r-plus-04-2024", 128_000)
    cohere_command_r_plus_08_2024 = Model("cohere:command-r-plus-08-2024", 128_000)
    cohere_command_r7b_12_2024 = Model("cohere:command-r7b-12-2024", 128_000)

    # DeepSeek models
    deepseek_deepseek_chat = Model("deepseek:deepseek-chat", 128_000)
    deepseek_deepseek_reasoner = Model("deepseek:deepseek-reasoner", 128_000)

    # Google models
    google_gla_gemini_2_0_flash = Model("google-gla:gemini-2.0-flash", 1_000_000)
    google_gla_gemini_2_0_flash_lite = Model(
        "google-gla:gemini-2.0-flash-lite", 1_000_000
    )
    google_gla_gemini_2_5_flash = Model("google-gla:gemini-2.5-flash", 1_000_000)
    google_gla_gemini_2_5_flash_lite = Model(
        "google-gla:gemini-2.5-flash-lite", 1_000_000
    )
    google_gla_gemini_2_5_pro = Model("google-gla:gemini-2.5-pro", 1_000_000)
    google_vertex_gemini_2_0_flash = Model("google-vertex:gemini-2.0-flash", 1_000_000)
    google_vertex_gemini_2_0_flash_lite = Model(
        "google-vertex:gemini-2.0-flash-lite", 1_000_000
    )
    google_vertex_gemini_2_5_flash = Model("google-vertex:gemini-2.5-flash", 1_000_000)
    google_vertex_gemini_2_5_flash_lite = Model(
        "google-vertex:gemini-2.5-flash-lite", 1_000_000
    )
    google_vertex_gemini_2_5_pro = Model("google-vertex:gemini-2.5-pro", 1_000_000)

    # Grok models
    grok_grok_4 = Model("grok:grok-4", 256_000)
    grok_grok_4_0709 = Model("grok:grok-4-0709", 256_000)
    grok_grok_3 = Model("grok:grok-3", 256_000)
    grok_grok_3_mini = Model("grok:grok-3-mini", 256_000)
    grok_grok_3_fast = Model("grok:grok-3-fast", 256_000)
    grok_grok_3_mini_fast = Model("grok:grok-3-mini-fast", 256_000)
    grok_grok_2_vision_1212 = Model("grok:grok-2-vision-1212", 128_000)
    grok_grok_2_image_1212 = Model("grok:grok-2-image-1212", 128_000)

    # Groq models
    groq_gemma2_9b_it = Model("groq:gemma2-9b-it", 8_192)
    groq_llama_3_3_70b_versatile = Model("groq:llama-3.3-70b-versatile", 128_000)
    groq_llama_3_1_8b_instant = Model("groq:llama-3.1-8b-instant", 128_000)
    groq_llama_guard_3_8b = Model("groq:llama-guard-3-8b", 8_192)
    groq_llama3_70b_8192 = Model("groq:llama3-70b-8192", 8_192)
    groq_llama3_8b_8192 = Model("groq:llama3-8b-8192", 8_192)
    groq_moonshotai_kimi_k2_instruct = Model(
        "groq:moonshotai/kimi-k2-instruct", 128_000
    )
    groq_qwen_qwq_32b = Model("groq:qwen-qwq-32b", 128_000)
    groq_mistral_saba_24b = Model("groq:mistral-saba-24b", 32_000)
    groq_qwen_2_5_coder_32b = Model("groq:qwen-2.5-coder-32b", 128_000)
    groq_qwen_2_5_32b = Model("groq:qwen-2.5-32b", 128_000)
    groq_deepseek_r1_distill_qwen_32b = Model(
        "groq:deepseek-r1-distill-qwen-32b", 128_000
    )
    groq_deepseek_r1_distill_llama_70b = Model(
        "groq:deepseek-r1-distill-llama-70b", 128_000
    )
    groq_llama_3_3_70b_specdec = Model("groq:llama-3.3-70b-specdec", 128_000)
    groq_llama_3_2_1b_preview = Model("groq:llama-3.2-1b-preview", 128_000)
    groq_llama_3_2_3b_preview = Model("groq:llama-3.2-3b-preview", 128_000)
    groq_llama_3_2_11b_vision_preview = Model(
        "groq:llama-3.2-11b-vision-preview", 128_000
    )
    groq_llama_3_2_90b_vision_preview = Model(
        "groq:llama-3.2-90b-vision-preview", 128_000
    )

    # Heroku models
    heroku_claude_3_5_haiku = Model("heroku:claude-3-5-haiku", 200_000)
    heroku_claude_3_5_sonnet_latest = Model("heroku:claude-3-5-sonnet-latest", 200_000)
    heroku_claude_3_7_sonnet = Model("heroku:claude-3-7-sonnet", 200_000)
    heroku_claude_4_sonnet = Model("heroku:claude-4-sonnet", 200_000)
    heroku_claude_3_haiku = Model("heroku:claude-3-haiku", 200_000)
    heroku_gpt_oss_120b = Model("heroku:gpt-oss-120b", 131_072)
    heroku_nova_lite = Model("heroku:nova-lite", 128_000)
    heroku_nova_pro = Model("heroku:nova-pro", 300_000)

    # HuggingFace models
    huggingface_qwen_qwq_32b = Model("huggingface:Qwen/QwQ-32B", 128_000)
    huggingface_qwen_qwen2_5_72b_instruct = Model(
        "huggingface:Qwen/Qwen2.5-72B-Instruct", 128_000
    )
    huggingface_qwen_qwen3_235b_a22b = Model(
        "huggingface:Qwen/Qwen3-235B-A22B", 128_000
    )
    huggingface_qwen_qwen3_32b = Model("huggingface:Qwen/Qwen3-32B", 128_000)
    huggingface_deepseek_ai_deepseek_r1 = Model(
        "huggingface:deepseek-ai/DeepSeek-R1", 128_000
    )
    huggingface_meta_llama_llama_3_3_70b_instruct = Model(
        "huggingface:meta-llama/Llama-3.3-70B-Instruct", 128_000
    )
    huggingface_meta_llama_llama_4_maverick_17b_128e_instruct = Model(
        "huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct", 10_000_000
    )
    huggingface_meta_llama_llama_4_scout_17b_16e_instruct = Model(
        "huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct", 10_000_000
    )

    # Mistral models
    mistral_codestral_latest = Model("mistral:codestral-latest", 32_000)
    mistral_mistral_large_latest = Model("mistral:mistral-large-latest", 128_000)
    mistral_mistral_moderation_latest = Model(
        "mistral:mistral-moderation-latest", 32_000
    )
    mistral_mistral_small_latest = Model("mistral:mistral-small-latest", 32_000)

    # MoonshotAI models
    moonshotai_moonshot_v1_8k = Model("moonshotai:moonshot-v1-8k", 8_000)
    moonshotai_moonshot_v1_32k = Model("moonshotai:moonshot-v1-32k", 32_000)
    moonshotai_moonshot_v1_128k = Model("moonshotai:moonshot-v1-128k", 128_000)
    moonshotai_moonshot_v1_8k_vision_preview = Model(
        "moonshotai:moonshot-v1-8k-vision-preview", 8_000
    )
    moonshotai_moonshot_v1_32k_vision_preview = Model(
        "moonshotai:moonshot-v1-32k-vision-preview", 32_000
    )
    moonshotai_moonshot_v1_128k_vision_preview = Model(
        "moonshotai:moonshot-v1-128k-vision-preview", 128_000
    )
    moonshotai_kimi_latest = Model("moonshotai:kimi-latest", 128_000)
    moonshotai_kimi_thinking_preview = Model(
        "moonshotai:kimi-thinking-preview", 128_000
    )
    moonshotai_kimi_k2_0711_preview = Model("moonshotai:kimi-k2-0711-preview", 128_000)

    # OpenAI models
    openai_chatgpt_4o_latest = Model("openai:chatgpt-4o-latest", 128_000, "gpt-4o")
    openai_codex_mini_latest = Model("openai:codex-mini-latest", 128_000, "")
    openai_gpt_3_5_turbo = Model("openai:gpt-3.5-turbo", 16_385, "gpt-3.5")
    openai_gpt_3_5_turbo_0125 = Model("openai:gpt-3.5-turbo-0125", 16_385, "gpt-3.5")
    openai_gpt_3_5_turbo_0301 = Model("openai:gpt-3.5-turbo-0301", 4_096, "gpt-3.5")
    openai_gpt_3_5_turbo_0613 = Model("openai:gpt-3.5-turbo-0613", 4_096, "gpt-3.5")
    openai_gpt_3_5_turbo_1106 = Model("openai:gpt-3.5-turbo-1106", 16_385, "gpt-3.5")
    openai_gpt_3_5_turbo_16k = Model("openai:gpt-3.5-turbo-16k", 16_385, "gpt-3.5")
    openai_gpt_3_5_turbo_16k_0613 = Model(
        "openai:gpt-3.5-turbo-16k-0613", 16_385, "gpt-3.5"
    )
    openai_gpt_4 = Model("openai:gpt-4", 8_192, "gpt-4")
    openai_gpt_4_0125_preview = Model("openai:gpt-4-0125-preview", 128_000, "gpt-4")
    openai_gpt_4_0314 = Model("openai:gpt-4-0314", 8_192, "gpt-4")
    openai_gpt_4_0613 = Model("openai:gpt-4-0613", 8_192, "gpt-4")
    openai_gpt_4_1106_preview = Model("openai:gpt-4-1106-preview", 128_000, "gpt-4")
    openai_gpt_4_32k = Model("openai:gpt-4-32k", 32_768, "gpt-4")
    openai_gpt_4_32k_0314 = Model("openai:gpt-4-32k-0314", 32_768, "gpt-4")
    openai_gpt_4_32k_0613 = Model("openai:gpt-4-32k-0613", 32_768, "gpt-4")
    openai_gpt_4_turbo = Model("openai:gpt-4-turbo", 128_000, "gpt-4")
    openai_gpt_4_turbo_2024_04_09 = Model(
        "openai:gpt-4-turbo-2024-04-09", 128_000, "gpt-4"
    )
    openai_gpt_4_turbo_preview = Model("openai:gpt-4-turbo-preview", 128_000, "gpt-4")
    openai_gpt_4_vision_preview = Model("openai:gpt-4-vision-preview", 128_000, "gpt-4")
    openai_gpt_4_1 = Model("openai:gpt-4.1", 1_000_000, "gpt-4.1")
    openai_gpt_4_1_2025_04_14 = Model("openai:gpt-4.1-2025-04-14", 1_000_000, "gpt-4.1")
    openai_gpt_4_1_mini = Model("openai:gpt-4.1-mini", 1_000_000, "gpt-4.1-mini")
    openai_gpt_4_1_mini_2025_04_14 = Model(
        "openai:gpt-4.1-mini-2025-04-14", 1_000_000, "gpt-4.1-mini"
    )
    openai_gpt_4_1_nano = Model("openai:gpt-4.1-nano", 1_000_000, "gpt-4.1-nano")
    openai_gpt_4_1_nano_2025_04_14 = Model(
        "openai:gpt-4.1-nano-2025-04-14", 1_000_000, "gpt-4.1-nano"
    )
    openai_gpt_4o = Model("openai:gpt-4o", 128_000, "gpt-4o")
    openai_gpt_4o_2024_05_13 = Model("openai:gpt-4o-2024-05-13", 128_000, "gpt-4o")
    openai_gpt_4o_2024_08_06 = Model("openai:gpt-4o-2024-08-06", 128_000, "gpt-4o")
    openai_gpt_4o_2024_11_20 = Model("openai:gpt-4o-2024-11-20", 128_000, "gpt-4o")
    openai_gpt_4o_mini = Model("openai:gpt-4o-mini", 128_000, "gpt-4o-mini")
    openai_gpt_4o_mini_2024_07_18 = Model(
        "openai:gpt-4o-mini-2024-07-18", 128_000, "gpt-4o-mini"
    )
    openai_gpt_4o_mini_search_preview = Model(
        "openai:gpt-4o-mini-search-preview", 128_000, "gpt-4o-mini"
    )
    openai_gpt_4o_mini_search_preview_2025_03_11 = Model(
        "openai:gpt-4o-mini-search-preview-2025-03-11", 128_000, "gpt-4o-mini"
    )
    openai_gpt_4o_search_preview = Model(
        "openai:gpt-4o-search-preview", 128_000, "gpt-4o-mini"
    )
    openai_gpt_4o_search_preview_2025_03_11 = Model(
        "openai:gpt-4o-search-preview-2025-03-11", 128_000, "gpt-4o-mini"
    )
    openai_gpt_5 = Model("openai:gpt-5", 400_000, "gpt-5")
    openai_gpt_5_2025_08_07 = Model("openai:gpt-5-2025-08-07", 400_000, "gpt-5")
    openai_o1 = Model("openai:o1", 200_000, "o1")
    openai_gpt_5_chat_latest = Model("openai:gpt-5-chat-latest", 400_000, "gpt-5")
    openai_o1_2024_12_17 = Model("openai:o1-2024-12-17", 200_000, "o1")
    openai_gpt_5_mini = Model("openai:gpt-5-mini", 400_000, "gpt-5-mini")
    openai_o1_mini = Model("openai:o1-mini", 128_000, "o1-mini")
    openai_gpt_5_mini_2025_08_07 = Model(
        "openai:gpt-5-mini-2025-08-07", 400_000, "gpt-5-mini"
    )
    openai_o1_mini_2024_09_12 = Model("openai:o1-mini-2024-09-12", 128_000, "o1-mini")
    openai_gpt_5_nano = Model("openai:gpt-5-nano", 400_000, "gpt-5-nano")
    openai_o1_preview = Model("openai:o1-preview", 128_000, "o1-preview")
    openai_gpt_5_nano_2025_08_07 = Model(
        "openai:gpt-5-nano-2025-08-07", 400_000, "gpt-5-nano"
    )
    openai_o1_preview_2024_09_12 = Model(
        "openai:o1-preview-2024-09-12", 128_000, "o1-preview"
    )
    openai_o1_pro = Model("openai:o1-pro", 128_000, "o1-pro")
    openai_o1_pro_2025_03_19 = Model("openai:o1-pro-2025-03-19", 128_000, "o1-pro")
    openai_o3 = Model("openai:o3", 200_000, "o3")
    openai_o3_2025_04_16 = Model("openai:o3-2025-04-16", 200_000, "o3")
    openai_o3_deep_research = Model("openai:o3-deep-research", 200_000, "o3")
    openai_o3_deep_research_2025_06_26 = Model(
        "openai:o3-deep-research-2025-06-26", 200_000, "o3"
    )
    openai_o3_mini = Model("openai:o3-mini", 200_000, "o3-mini")
    openai_o3_mini_2025_01_31 = Model("openai:o3-mini-2025-01-31", 200_000, "o3-mini")
    openai_o4_mini = Model("openai:o4-mini", 200_000, "o4-mini")
    openai_o4_mini_2025_04_16 = Model("openai:o4-mini-2025-04-16", 200_000, "o4-mini")
    openai_o4_mini_deep_research = Model(
        "openai:o4-mini-deep-research", 200_000, "o4-mini"
    )
    openai_o4_mini_deep_research_2025_06_26 = Model(
        "openai:o4-mini-deep-research-2025-06-26", 200_000, "o4-mini"
    )
    openai_o3_pro = Model("openai:o3-pro", 200_000, "o3")
    openai_o3_pro_2025_06_10 = Model("openai:o3-pro-2025-06-10", 200_000, "o3")

    # Test model
    test = Model("test", 1_000_000)
