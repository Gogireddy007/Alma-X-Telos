# One command reruns the whole validation:  make validate RUN=<run folder name> CKPT=<ema ckpt> LAST=<last.pt>
PY = .venv/bin/python
RUN ?= $(notdir $(shell cat validation_runs/CURRENT))
CKPT ?= runs/alma_wm1/ema_8000.pt
LAST ?= runs/alma_wm1/last.pt
export VALRUN = validation_runs/$(RUN)

validate: predict static grade report

suites:
	$(PY) validate/build_suite.py sim real
predict:
	$(PY) validate/predict.py --ckpt $(CKPT)
static:
	$(PY) validate/static_tests.py --ckpt $(CKPT)
grade:
	$(PY) validate/grade.py --ckpt $(CKPT) --last $(LAST)
report:
	$(PY) validate/make_report.py

.PHONY: validate suites predict static grade report
