import os
import json
import torch
import fnmatch

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoModel

from lm_eval.arguments import EvalArguments
from lm_eval.evaluator import Evaluator
from lm_eval.tasks import ALL_TASKS

from utils.model import Classifier

def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="bigcode/starcoder",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="for LLaMA, the local path of llama ckpts is needed",
    )
    parser.add_argument(
        "--proxy_model",
        default=None,
        help="for SWEET, the local path of surrogate ckpts is needed",
    )
    parser.add_argument(
        "--embedding_model_path",
        default=None,
        help="for IE, the local path of embedding ckpts is needed",
    )
    parser.add_argument(
        "--classifier_ckpt_path",
        default=None,
        help="for IE, the local path of classifier ckpts is needed",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--wllm",
        action="store_true",
        help="Whether to use Watermark for Language Model",
    ) 
    parser.add_argument(
        "--ie",
        action="store_true",
        help="Whether to use Information Embedding",
    )
    parser.add_argument(
        "--sweet",
        action="store_true",
        help="Whether to use SWEET",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for WLLM,SWEET",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
        help="Delta for WLLM,SWEET",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Entropy threshold for SWEET",
    )
    parser.add_argument(
        "--key_length",
        type=int,
        default=512,
        help="key length for EXP-edit",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="block size for EXP-edit",
    )
    parser.add_argument(
        "--n_detection",
        type=int,
        default=1,
        help="the number of code generated for detection among n_samples",
    )
    parser.add_argument(
        "--detect_human_code",
        action="store_true",
        help="Detect human code, NOT generated code",
    )
    return parser.parse_args()

def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def main():
    args = parse_args()
    
    assert args.task is not None
    task_name = pattern_match([args.task], ALL_TASKS)[0]

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f'Selected Task: {task_name}')
    
    os.makedirs(args.outputs_dir, exist_ok=True)
    args.save_generations_path = os.path.join(args.outputs_dir, args.save_generations_path)
    args.metric_output_path = os.path.join(args.outputs_dir, args.metric_output_path)

    results = {}

    if args.model != "meta-llama/Llama-2":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            cache_dir=args.model_path,
            truncation_side="left",
            padding_side="right",
            #use_fast=False,
        )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    tokenizer.pad_token = tokenizer.eos_token
    
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_generations_path or args.detect_human_code:

        assert not (args.load_generations_path and args.detect_human_code), \
            "Choose only one between 1) detecting generated code vs 2) human code"

        # here we don't generate code but only evaluate previously computed generations
        if accelerator.is_main_process:
            print("evaluation only mode")

            if args.sweet:
                if args.entropy_file_path:
                    model = None
                else:
                    if args.model != "meta-llama/Llama-2":
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model,
                            revision=args.revision,
                            torch_dtype=dict_precisions[args.precision],
                            trust_remote_code=args.trust_remote_code,
                            device_map={"": device}
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model,
                            cache_dir=args.model_path,
                            torch_dtype="auto",
                            device_map={"": device}
                        )

            elif args.ie:
                embedding_model = AutoModel.from_pretrained(args.embedding_model_path, torch_dtype=torch.float16, device_map={'':device}).eval()
                embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
                classifier = Classifier(input_dim = embedding_model.config.hidden_size)
                classifier.load_state_dict(torch.load(args.classifier_ckpt_path, map_location=device, weights_only=True))
                classifier.eval()
                classifier.to(device)
                model = None

            else:
                model = None
        
        if args.ie:
            evaluator = Evaluator(
                accelerator = accelerator,
                tokenizer = tokenizer,
                args = args,
                model = model,
                embed_model = embedding_model,
                embed_tokenizer = embedding_tokenizer,
                classifier = classifier
            )
        else:
            evaluator = Evaluator(accelerator, tokenizer, args, model)

        results[task_name] = evaluator.evaluate(task_name)

    else:
        print(f"Loading tokenizer and model (in {args.precision})")
        if args.model != "meta-llama/Llama-2":
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dict_precisions[args.precision],
                trust_remote_code=args.trust_remote_code,
                device_map={"": device}
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                cache_dir=args.model_path,
                torch_dtype="auto",
            )
        # here we generate code and save it (evaluation is optional but True by default)
        if args.ie:
            embedding_model = AutoModel.from_pretrained(args.embedding_model_path, torch_dtype=torch.float16, device_map='auto').eval()
            embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
            classifier = Classifier(input_dim = embedding_model.config.hidden_size)
            classifier.load_state_dict(torch.load(args.classifier_ckpt_path, map_location=device, weights_only=True))
            classifier.to(device)
            classifier.eval()
            evaluator = Evaluator(accelerator,  tokenizer, args, model, embedding_model, embedding_tokenizer, classifier)
        else:
            evaluator = Evaluator(accelerator, tokenizer, args, model)

        if args.generation_only:
            assert not args.detect_human_code, \
                "Choose only one between 1) generating code vs 2) detecting human code"

            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(task_name, args.start_index, args.end_index)
            if accelerator.is_main_process:
                with open(args.save_generations_path, "w") as fp:
                    json.dump(generations, fp)
                    print(f"generations were saved at {args.save_generations_path}")
                if args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")
        else:
            results[task_name] = evaluator.evaluate(task_name)

    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        with open(args.metric_output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()