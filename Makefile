.PHONY: dashboard dashboard-docker

# Copy benchmark DB into Evidence sources and start dev server
dashboard:
	@mkdir -p docs/sources/bench
	cp benchmarks/results/results.db docs/sources/bench/results.db
	cd docs && npm run sources && npm run dev

# Build and run dashboard in a container (podman-compatible, :z for SELinux)
DASHBOARD_IMAGE := hip-rwkv-dashboard
DASHBOARD_PORT  ?= 3000

dashboard-docker:
	@mkdir -p docs/sources/bench
	cp benchmarks/results/results.db docs/sources/bench/results.db
	podman build -t $(DASHBOARD_IMAGE) docs/
	@echo "Starting dashboard on http://localhost:$(DASHBOARD_PORT)"
	podman run --rm -it -p $(DASHBOARD_PORT):3000 $(DASHBOARD_IMAGE)
