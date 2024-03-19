from transformers import AutoTokenizer
import torch
from modeling.modeling_moe import MoEForCausalLM
from modeling.configuration_moe import MoEConfig



def generate(tokenizer, model, text):
    inputs = [text]
    tokens = tokenizer(inputs,return_tensors="pt")
    input_ids = tokens.input_ids.cuda()
    generate_ids = model.generate(inputs=input_ids,
                num_beams=1, 
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,top_p=0.9, temperature=1.0, do_sample=True)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    response = [outputs[i][len(inputs[i]):] for i in range(len(outputs))][0]
    return response    
    
    

if __name__ == "__main__":
    model_path = 'path_to_dynamicmoe_moedel'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.unk_token

    model_config = MoEConfig.from_pretrained(model_path,trust_remote_code=True)
    model = MoEForCausalLM.from_pretrained(
        model_path,
        from_tf=False,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).cuda()    
    model.eval() 

    response = generate(tokenizer, model, 'The highest mountain in the world is')
    print(response)
    
