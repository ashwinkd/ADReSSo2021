models = ['Bert', 'Roberta']
_ = [print(f"[{i + 1}] {m}") for i, m in enumerate(models)]
model_num = int(input("Choose model: ")) - 1
if model_num not in range(len(models)):
    raise Exception("Incorrect model chosen.")

model_name = models[model_num]
