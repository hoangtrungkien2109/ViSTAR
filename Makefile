.PHONY: start stop restart logs stopall startall restartall

start:
	@echo "Starting all services..."
	@bash scripts/start.sh

stop:
	@echo "Stopping all services..."
	@bash scripts/stop.sh

restart: stop start

logs:
	@echo "Tailing logs..."
	@tail -f logs/*.log

startall:
	@echo "Starting all services..."
	@bash scripts/init.sh
	@bash scripts/start.sh

stopall:
	@echo "Stopping all services..."
	@bash scripts/stop.sh
	@bash scripts/dockerdown.sh

restartall: stopall startall