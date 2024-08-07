import os, torch  # os와 torch 모듈을 불러옵니다.

from datasets import load_dataset  # datasets 모듈에서 load_dataset 함수를 불러옵니다.
from trl import SFTTrainer, setup_chat_format  # trl 모듈에서 SFTTrainer와 setup_chat_format 함수를 불러옵니다.

from peft import (
    LoraConfig,  # peft 모듈에서 LoraConfig 클래스를 불러옵니다.
    PeftModel,  # peft 모듈에서 PeftModel 클래스를 불러옵니다.
    prepare_model_for_kbit_training,  # peft 모듈에서 prepare_model_for_kbit_training 함수를 불러옵니다.
    get_peft_model,  # peft 모듈에서 get_peft_model 함수를 불러옵니다.
)

from transformers import (
    AutoModelForCausalLM,  # transformers 모듈에서 AutoModelForCausalLM 클래스를 불러옵니다.
    AutoTokenizer,  # transformers 모듈에서 AutoTokenizer 클래스를 불러옵니다.
    BitsAndBytesConfig,  # transformers 모듈에서 BitsAndBytesConfig 클래스를 불러옵니다.
    HfArgumentParser,  # transformers 모듈에서 HfArgumentParser 클래스를 불러옵니다.
    TrainingArguments,  # transformers 모듈에서 TrainingArguments 클래스를 불러옵니다.
    pipeline,  # transformers 모듈에서 pipeline 함수를 불러옵니다.
)


if __name__ == '__main__':
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # MLP-KTLim/llama-3-Korean-Bllossom-8B 모델의 이름을 설정합니다.
    dataset_name = "beomi/KoAlpaca-v1.1a"  # 학습에 사용할 데이터셋의 이름을 설정합니다.
    finetuned_model = "llama-3-8b-chat-doctor"  # 파인튜닝된 모델의 이름을 설정합니다.

    torch_dtype = torch.float16  # 모델의 데이터 타입을 설정합니다.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 모델을 4비트로 로드합니다.
        bnb_4bit_quant_type="nf4",  # 4비트 양자화 유형을 설정합니다.
        bnb_4bit_compute_dtype=torch_dtype,  # 4비트 계산 데이터 타입을 설정합니다.
        bnb_4bit_use_double_quant=True,  # 이중 양자화를 사용합니다.
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # 사전 학습된 모델을 로드합니다.
        quantization_config=bnb_config,  # 양자화 설정을 적용합니다.
        device_map="auto",  # 자동으로 장치를 매핑합니다.
        cache_dir="/data"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 사전 학습된 모델의 토크나이저를 로드합니다.
    model, tokenizer = setup_chat_format(model, tokenizer)  # 채팅 형식에 맞게 모델과 토크나이저를 설정합니다.

    peft_config = LoraConfig(
        r=4,  # Low-rank matrix의 rank를 설정합니다.
        lora_alpha=8,  # LoRA의 알파 값을 설정합니다.
        task_type="CAUSAL_LM",  # 작업 유형을 설정합니다.
        lora_dropout=0.1,  # 드롭아웃 비율을 설정합니다.
        target_modules=['k_proj', 'q_proj', 'v_proj']  # LoRA를 적용할 모듈을 지정합니다.
    )
    model = get_peft_model(model, peft_config)  # PEFT 설정을 적용하여 모델을 가져옵니다.

    dataset = load_dataset(dataset_name, split="all")  # 데이터셋을 모두 불러옵니다.


    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["instruction"]},  # instruction을 사용자 발화를 적용합니다.
                    {"role": "assistant", "content": row["output"]}]  # output을 어시스턴트 발화를 적용합니다.
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)  # 채팅 템플릿을 적용하여 텍스트로 변환합니다.
        return row


    dataset = dataset.map(
        format_chat_template  # 각 데이터셋 행에 채팅 템플릿을 적용합니다.
    )

    print(dataset['instruction'][0])  # 첫 번째 instruction을 출력합니다.
    print(dataset['output'][0])  # 첫 번째 output을 출력합니다.

    dataset = dataset.train_test_split(test_size=0.1)  # 학습 데이터와 평가용 데이터를 분리합니다.
    print(dataset)

    training_arguments = TrainingArguments(
        output_dir=finetuned_model,  # 파인튜닝된 모델을 저장할 디렉토리를 설정합니다.
        per_device_train_batch_size=1,  # 훈련 시 각 장치에서 사용할 배치 크기를 설정합니다.
        # per_device_eval_batch_size=1,  # 평가 시 각 장치에서 사용할 배치 크기를 설정합니다.
        gradient_accumulation_steps=32,  # 그래디언트 누적 단계를 설정하여 메모리 사용을 줄입니다.
        num_train_epochs=4,  # 훈련할 에포크 수를 설정합니다.
        # evaluation_strategy="steps",  # 평가 전략을 단계별로 설정합니다.
        # eval_steps=0.2,  # 평가를 수행할 단계 수를 설정합니다.
        warmup_steps=200,  # 학습률을 선형으로 증가시키는 워밍업 단계 수를 설정합니다.
        optim="paged_adamw_32bit",  # 최적화 알고리즘으로 PagedAdamW를 32비트로 설정합니다.
        learning_rate=2e-5,  # 초기 학습률을 설정합니다.
        fp16=False,  # 16비트 부동 소수점 사용 여부를 설정합니다.
        bf16=False,  # bfloat16 사용 여부를 설정합니다.
        group_by_length=True,  # 배치를 길이별로 그룹화하여 패딩을 최소화합니다.
        report_to=[], # 옵션을 빈 리스트로 설정
    )

    trainer = SFTTrainer(
        model=model,  # 훈련할 모델을 설정합니다.
        train_dataset=dataset["train"],  # 훈련에 사용할 데이터셋을 설정합니다.
        # eval_dataset=dataset["test"],  # 평가에 사용할 데이터셋을 설정합니다.
        peft_config=peft_config,  # PEFT 설정을 적용합니다.
        max_seq_length=256,  # 최대 시퀀스 길이를 설정합니다.
        dataset_text_field="text",  # 텍스트 필드 이름을 설정합니다.
        tokenizer=tokenizer,  # 토크나이저를 설정합니다.
        args=training_arguments,  # 훈련을 위한 하이퍼파라미터를 설정합니다.
        packing=False,  # 패킹 사용 여부를 설정합니다.
    )

    trainer.train()  # 모델 학습


