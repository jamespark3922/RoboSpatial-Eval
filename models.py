# models.py

def load_robopoint_model(model_path=None):
    # Robopoint-specific imports
    from robopoint.model.builder import load_pretrained_model
    from robopoint.utils import disable_torch_init
    from robopoint.mm_utils import get_model_name_from_path

    # Disable torch initialization for faster loading
    disable_torch_init()

    # Use provided model path or default
    if model_path is None:
        model_path = 'wentao-yuan/robopoint-v1-vicuna-v1.5-13b'
    model_base = None  # Update if necessary

    # Load model name
    model_name = get_model_name_from_path(model_path)

    # Load tokenizer, model, image_processor, context_len
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    # Prepare model kwargs
    model_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
    }

    return model_kwargs

def load_spatialvlm_model(model_path=None):
    # SpatialVLM-specific imports
    import torch
    from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration

    attn_implementation = "flash_attention_2"
    if model_path is None:
        model_path = "remyxai/SpaceMantis"
    processor = MLlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation
    )

    generation_kwargs = {
        "max_new_tokens": 1024,
        "num_beams": 1,
        "do_sample": False
    }

    model_kwargs = {
        "model": model,
        "processor": processor,
        "generation_kwargs": generation_kwargs
    }
    return model_kwargs

def load_llava_next_model(model_path=None):
    from llava.model.builder import load_pretrained_model
    device_map = "cuda"
    if model_path is None:
        model_path = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, device_map=device_map, attn_implementation=None
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "tokenizer": tokenizer, 'image_processor': image_processor}
    return model_kwargs


def load_molmo_model(model_path=None):
    from transformers import AutoModelForCausalLM, AutoProcessor
    if model_path is None:
        model_path = "allenai/Molmo-7B-D-0924"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model.eval()
    model.tie_weights()
    model_kwargs = {"model": model, "processor": processor}
    return model_kwargs


def _load_molmo_image_text_model(model_path, default_model_path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = model_path or default_model_path
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return {
        "model": model,
        "processor": processor,
        "model_path": model_path,
        "max_new_tokens": 200 if "MolmoPoint" in model_path else 128,
    }


def load_molmopoint_model(model_path=None):
    return _load_molmo_image_text_model(model_path, "allenai/MolmoPoint-8B")


def load_molmo2_model(model_path=None):
    return _load_molmo_image_text_model(model_path, "allenai/Molmo2-8B")


def _first_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _move_batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def _format_robospatial_pointing_prompt(question):
    import re

    prompt = str(question).strip()
    prompt = re.sub(
        r"\s*Your answer should be formatted as a list of tuples.*$",
        "",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    prompt = re.sub(
        r"\bPinpoint several points within\b",
        "Point to",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(
        r"\bPinpoint a point within\b",
        "Point to",
        prompt,
        flags=re.IGNORECASE,
    )
    return prompt.strip()


def _format_robospatial_prompt(question, category=None):
    if str(category).lower() in {"configuration", "compatibility"}:
        return f"{question}\nAnswer with only yes or no."
    return _format_robospatial_pointing_prompt(question)


def _decode_generated_text(output_ids, inputs, processor):
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = output_ids[:, input_len:]
    return processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _normalize_point_text_for_robospatial(text):
    import re

    points = []
    number_any = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"

    def add_point(x, y):
        x = float(x)
        y = float(y)
        scale = 1.0
        max_coord = max(abs(x), abs(y))
        if max_coord > 100.0:
            scale = 1000.0
        elif max_coord > 1.0:
            scale = 100.0
        points.append((x / scale, y / scale))

    def add_coord_sequence(coord_text):
        nums = [float(n) for n in re.findall(number_any, coord_text)]
        if len(nums) < 2:
            return
        # Molmo2 often emits numbered points: id,x,y,id,x,y,...
        if len(nums) % 3 == 0 and all(nums[i].is_integer() for i in range(0, len(nums), 3)):
            for i in range(0, len(nums), 3):
                add_point(nums[i + 1], nums[i + 2])
                if len(points) >= 2:
                    return
            return
        # Some outputs include a leading "1 1" prefix before x,y.
        if len(nums) == 4 and nums[0].is_integer() and nums[1].is_integer():
            add_point(nums[2], nums[3])
            return
        for i in range(0, len(nums) - 1, 2):
            add_point(nums[i], nums[i + 1])
            if len(points) >= 2:
                return

    for match in re.finditer(r'<points?\s+coords="([^"]+)"', text):
        add_coord_sequence(match.group(1))
        if len(points) >= 2:
            break

    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    patterns = [
        rf"Click\(\s*{number}\s*,\s*{number}\s*\)",
        rf"[\(\[]\s*{number}\s*[, ]\s*{number}\s*[\)\]]",
        rf'x\d*="\s*{number}"\s+y\d*="\s*{number}"',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                add_point(match.group(1), match.group(2))
            except ValueError:
                continue
            if len(points) >= 2:
                break
        if points:
            break

    if not points:
        return text
    return str([(round(x, 4), round(y, 4)) for x, y in points])


def _point_rows_to_normalized_text(points, image_size):
    width, height = image_size
    normalized_points = []
    for point in points:
        if len(point) < 2:
            continue
        x = float(point[-2])
        y = float(point[-1])
        normalized_points.append((round(x / width, 4), round(y / height, 4)))
        if len(normalized_points) >= 2:
            break
    return str(normalized_points) if normalized_points else None


def _generate_molmo_image_text_answer(question, image_path, kwargs):
    import torch
    from PIL import Image

    model = kwargs["model"]
    processor = kwargs["processor"]
    category = kwargs.get("category")
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    image = Image.open(image_path).convert("RGB")
    prompt = _format_robospatial_prompt(question, category=category)
    if kwargs.get("print_prompts"):
        print("\n" + "=" * 80)
        print(f"ROBOSPATIAL PROMPT category={category} image={image_path}")
        print("-" * 80)
        print(prompt)
        print("=" * 80)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    use_native_pointing = hasattr(model, "extract_image_points") and hasattr(model, "build_logit_processor_from_inputs")
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if use_native_pointing:
        template_kwargs["padding"] = True
        template_kwargs["return_pointing_metadata"] = True

    inputs = processor.apply_chat_template(messages, **template_kwargs)
    metadata = inputs.pop("metadata", None)
    inputs = _move_batch_to_device(inputs, _first_model_device(model))

    device = _first_model_device(model)
    autocast_enabled = getattr(device, "type", str(device)).startswith("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        generate_kwargs = {}
        if use_native_pointing:
            generate_kwargs["logits_processor"] = model.build_logit_processor_from_inputs(inputs)
        output_ids = model.generate(
            **inputs,
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if use_native_pointing and metadata is not None:
        generated_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        if hasattr(processor, "post_process_image_text_to_text"):
            generated_text = processor.post_process_image_text_to_text(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            generated_text = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
        points = model.extract_image_points(
            generated_text,
            metadata["token_pooling"],
            metadata["subpatch_mapping"],
            metadata["image_sizes"],
        )
        normalized_text = _point_rows_to_normalized_text(points, image.size)
        if normalized_text:
            return normalized_text
    else:
        generated_text = _decode_generated_text(output_ids, inputs, processor)
    return _normalize_point_text_for_robospatial(generated_text)


def run_molmopoint(question, image_path, depth_path, kwargs):
    return _generate_molmo_image_text_answer(question, image_path, kwargs)


def run_molmo2(question, image_path, depth_path, kwargs):
    return _generate_molmo_image_text_answer(question, image_path, kwargs)

def load_gpt_model():
    return {}

def load_qwen2vl_model(model_path=None):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    if model_path is None:
        model_path = "Qwen/Qwen2-VL-7B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    model_kwargs = {
        "model": model,
        "processor": processor
    }
    return model_kwargs


def run_robopoint(question, image_path, depth_path, kwargs):
    # Robopoint-specific imports
    import torch
    from PIL import Image
    from robopoint.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    )
    from robopoint.conversation import conv_templates
    from robopoint.mm_utils import tokenizer_image_token, process_images

    # Extract necessary components from kwargs
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    image_processor = kwargs["image_processor"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process the question
    if DEFAULT_IMAGE_TOKEN not in question:
        if getattr(model.config, 'mm_use_im_start_end', False):
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        question = question.split('\n', 1)[1]

    # Conversation mode
    conv_mode = "llava_v1"  # Update if necessary
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().to(device)

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,  # Set to True if you want sampling
            temperature=0.2,  # Adjust as needed
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_spatialvlm(question, image_path, depth_path, kwargs):
    # SpatialVLM-specific imports
    from PIL import Image
    from mantis.models.mllava import chat_mllava

    model = kwargs["model"]
    processor = kwargs["processor"]
    generation_kwargs = kwargs["generation_kwargs"]

    # Load the image
    image = Image.open(image_path).convert("RGB")
    images = [image]

    # Run the inference
    response, history = chat_mllava(question, images, model, processor, **generation_kwargs)
    return response.strip()


def run_llava_next(question, image_path, depth_path, kwargs):
    # LLAVA-specific imports
    from PIL import Image
    import torch
    import copy
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    image_processor = kwargs["image_processor"]
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    device = "cuda"

    image = Image.open(image_path)
    # Depth is provided for all examples; most LLaVA-Next checkpoints won't use it.
    # Model builders can adapt this runner to incorporate depth if their model supports it.
    _depth_image = None
    if depth_path:
        try:
            _depth_image = Image.open(depth_path)
        except Exception:
            _depth_image = None

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"  # Use the correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0].strip()


def run_molmo(question, image_path, depth_path, kwargs):
    """
    Run the Molmo model using the generate_answer function.
    """
    def generate_answer(image_path, question, model, processor, **kwargs):
        from PIL import Image
        from transformers import GenerationConfig

        # Process the image and text
        inputs = processor.process(
            images=[Image.open(image_path)],
            text=question
        )

        # Move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Create a GenerationConfig and update it with any additional kwargs
        generation_config = GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>")
        for key, value in kwargs.items():
            setattr(generation_config, key, value)

        # Generate output
        output = model.generate_from_batch(
            inputs,
            generation_config,
            tokenizer=processor.tokenizer
        )

        # Extract generated tokens and decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    model = kwargs["model"]
    processor = kwargs["processor"]
    # Remove model and processor from kwargs to avoid conflicts
    generation_kwargs = {k: v for k, v in kwargs.items() if k not in ["model", "processor"]}
    generated_text = generate_answer(image_path, question, model, processor, **generation_kwargs)
    return generated_text


def send_question_to_openai(question, image_base64):
    """
    Send a question and base64 encoded image to the GPT-4 model and get the response.
    """
    from openai import OpenAI
    # Set your OpenAI API key if needed
    # openai.api_key = 'your-api-key'

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ]
    )

    return response.output[0].content[0].text


def run_qwen2vl(question, image_path, depth_path, kwargs):
    """
    Use the Qwen2-VL model to answer the question about the given image.
    We'll build a 'messages' format, apply the chat template, and then generate.
    """
    import torch
    from PIL import Image

    model = kwargs["model"]
    processor = kwargs["processor"]

    device = "cuda"  # or use model.device

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    pil_image = Image.open(image_path).convert("RGB")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()
