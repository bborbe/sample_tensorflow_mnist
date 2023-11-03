
deps:
	pip install -r requirements.txt

run:
	python run_basic.py

train:
	python run_train_model_with_checkpoints.py

evaluate:
	python run_load_model.py
