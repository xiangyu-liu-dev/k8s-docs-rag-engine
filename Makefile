REF = release-1.33

.PHONY: init checkout container-render ingest-html ingest-md build-index vllm-start vllm-stop uvicorn-start uvicorn-stop eval-retrieval benchmark-baseline benchmark-vllm benchmark-all start stop restart

init:
	git submodule update --init --recursive

checkout:
	cd data/website && git fetch --all && git checkout $(REF)

CONTAINER_IMAGE = gcr.io/k8s-staging-sig-docs/k8s-website-hugo:v0.133.0-9fd193256b90
CONTAINER_HUGO_MOUNTS = \
	--mount type=bind,source=./.git,target=/src/.git,readonly \
	--mount type=bind,source=./archetypes,target=/src/archetypes,readonly \
	--mount type=bind,source=./assets,target=/src/assets,readonly \
	--mount type=bind,source=./content,target=/src/content,readonly \
	--mount type=bind,source=./data,target=/src/data,readonly \
	--mount type=bind,source=./i18n,target=/src/i18n,readonly \
	--mount type=bind,source=./layouts,target=/src/layouts,readonly \
	--mount type=bind,source=./static,target=/src/static,readonly \
	--mount type=tmpfs,destination=/tmp,tmpfs-mode=01777 \
	--mount type=bind,source=./hugo.toml,target=/src/hugo.toml,readonly \
	--mount type=bind,source=./public,target=/src/public \

container-render:
	cd data/website && \
	mkdir -p public && \
	docker run --rm \
	$(CONTAINER_HUGO_MOUNTS) $(CONTAINER_IMAGE) hugo --destination /src/public --enableGitInfo=false --minify --cleanDestinationDir --environment development --renderSegments=en --noBuildLock
	rm -rf data/rendered
	mkdir -p data/rendered
	cp -r data/website/public/docs data/rendered/
	rm -rf data/website/public

ingest-html:
	mkdir -p data/processed
	uv run python -m ingest.html_ingest.parse_html \
	  --html-root data/rendered/docs \
	  --out data/processed/chunks_html.jsonl \
	  --ref $(REF)

ingest-md:
	uv run python -m ingest.md_ingest.build_breadcrumbs \
	  --docs-root data/website/content/en/docs \
	  --out data/processed/breadcrumbs.jsonl
	uv run python -m ingest.md_ingest.parse_md \
	  --md-root data/website/content/en/docs \
	  --breadcrumbs data/processed/breadcrumbs.jsonl \
	  --out data/processed/chunks_md.jsonl \
	  --ref $(REF)

build-index:
	uv run python -m rag.build_index \
	  --chunks data/processed/chunks_html.jsonl \
	  --out data/vector_index

vllm-start:
	@mkdir -p logs
	@echo "Starting vLLM server..."
	@@NO_COLOR=1 uv run python -m vllm.entrypoints.openai.api_server \
		--model Qwen/Qwen2.5-7B-Instruct \
		--dtype bfloat16 \
		--max-model-len 8192 \
		--gpu-memory-utilization 0.75 \
		--enable-prefix-caching \
		--port 8100 \
		> logs/vllm.log 2>&1 &
	@echo "Waiting for server to be ready..."
	@for i in {1..120}; do \
		if curl -s http://localhost:8100/health > /dev/null 2>&1; then \
			echo "✓ vLLM server ready on port 8100"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "✗ vLLM server failed to start within 120 seconds"; \
	echo "Check logs/vllm.log for details"; \
	exit 1

vllm-stop:
	@echo "Stopping vLLM server..."
	@pkill -f "vllm.entrypoints.openai.api_server" || true

uvicorn-start: vllm-start
	@echo "Starting uvicorn server..."
	@uv run uvicorn app.server:app \
		--host 0.0.0.0 \
		--port 8000 \
		> logs/uvicorn.log 2>&1 &
	@echo "Waiting for uvicorn to be ready..."
	@for i in {1..30}; do \
		if curl -s http://localhost:8000/health > /dev/null 2>&1; then \
			echo "✓ Uvicorn server ready on port 8000"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "✗ Uvicorn server failed to start within 30 seconds"; \
	echo "Check logs/uvicorn.log for details"; \
	exit 1

uvicorn-stop:
	@echo "Stopping uvicorn server..."
	@pkill -f "uvicorn app.server:app" || true

eval-retrieval:
	uv run python -m eval.eval_retrieval
	@make stop
	uv run python -m eval.generate_answers_transformer
	@make vllm-start
	uv run python -m eval.generate_answers_vllm
	@make stop
	uv run python -m eval.judge_answers

CONCURRENCY_LEVELS = 1 5 10 20

benchmark-baseline:
	@make stop
	uv run python -m bench.bench_baseline

benchmark-vllm:
	@echo "========== Direct sequential =========="
	make vllm-start
	uv run python -m bench.bench_vllm --direct
	make stop
	sleep 10
	@for n in $(CONCURRENCY_LEVELS); do \
		echo "\n========== Concurrent=$$n =========="; \
		make start; \
		uv run python -m bench.bench_vllm -n $$n; \
		make stop; \
		sleep 10; \
	done

benchmark-all: benchmark-baseline benchmark-vllm
	uv run python -m bench.summarize

start: uvicorn-start
	@echo "✓ All servers started"

stop:
	@echo "Stopping uvicorn..."
	-@pkill -f "uvicorn app.server:app" 2>/dev/null
	@echo "Stopping vllm..."
	-@pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
	@echo "Done."

restart: stop start