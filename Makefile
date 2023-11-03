
deps:
	pip install -r requirements.txt

run:
	python run_basic.py --epochs=10

train:
	python run_train_model_with_checkpoints.py --epochs=10

evaluate:
	python run_load_model.py
