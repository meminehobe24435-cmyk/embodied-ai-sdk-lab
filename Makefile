PY ?= python
.PHONY: train bench infer test ui clean
train:
	@$(PY) cli.py train --epochs 60
bench:
	@$(PY) cli.py bench --batch 8 --repeat 20
infer:
	@$(PY) cli.py infer --frames 8
ui:
	@$(PY) cli.py infer --frames 8 --save
test:
	@$(PY) -m unittest discover -s tests -v
clean:
	-@cmd /c rmdir /s /q artifacts 2>nul
