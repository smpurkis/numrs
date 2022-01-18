#!/bin/bash

wasm-pack build --target web --release
cat ./custom_js_glue/numrs_glue.js >> ./pkg/numrs.js


