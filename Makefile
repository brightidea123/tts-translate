SWIFTC ?= swiftc
SWIFT_MODULE_CACHE ?= /tmp/tts-translate-swift-module-cache
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
NODE ?= node

.PHONY: build-native check-native check-python check-tools check-web check doctor clean-native

build-native: apple_vpio_bridge

apple_vpio_bridge: apple_vpio_bridge.swift
	$(SWIFTC) -module-cache-path $(SWIFT_MODULE_CACHE) $< -framework AVFoundation -o $@

check-native: build-native
	./apple_vpio_bridge --self-test

check-python:
	$(PYTHON) -c 'import sys; sys.exit("Python 3.10+ required; activate the project virtual environment") if sys.version_info < (3, 10) else None'
	$(PYTHON) tts-transcribe.py --self-test

check-tools:
	$(PYTHON) -m unittest discover -s tests

check-web:
	$(NODE) --check web/app.js
	$(NODE) --check web/ui-interactions.js
	$(NODE) scripts/test_web_ui.mjs

doctor:
	$(PYTHON) scripts/doctor.py --devices

check: check-python check-tools check-native

clean-native:
	rm -f apple_vpio_bridge
