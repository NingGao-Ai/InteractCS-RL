"""
User-Agent Customer Service Simulation Framework - Main Entry Point
"""
import importlib
import pkgutil
import logging
import os
from datetime import datetime
from pathlib import Path
from core.registry import init_registry
logger = logging.getLogger(__name__)


class Application:
    """Application Launcher"""
    
    def __init__(self, config_path: str, cli_args: list = None):
        self.config_path = config_path
        self.registry = None
        self.cli_args = cli_args
        
        # Initialize registry (including loading config and applying parameter overrides)
        self.registry = init_registry(config_path, cli_args=cli_args)
        
        # Get config info from registry for logging setup
        config_loader = self.registry._config_loader
        app_config = config_loader.config.get('app', {})
        framework_root = os.path.dirname(os.path.abspath(__file__))
        
        
        # Set up log directory
        logger_dir = app_config.get('logger_dir', 'logger')
        if not os.path.isabs(logger_dir):
            logger_dir = os.path.join(framework_root, logger_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d")
        self.logger_dir = os.path.join(logger_dir, timestamp)
        Path(self.logger_dir).mkdir(parents=True, exist_ok=True)
        
        self.log_level = app_config.get('log_level', 'INFO')
        self.log_to_file = app_config.get('log_to_file', True)
    
    def initialize(self):
        logger.info("Starting application initialization")
        self._auto_import_components()
        self.registry.init_component()
        logger.info("Application initialization complete")
    
    def _auto_import_components(self):
        for pkg_name in ['agents', 'session']:
            try:
                pkg = importlib.import_module(pkg_name)
                pkg_path = pkg.__path__[0] if hasattr(pkg, '__path__') else None
                
                if pkg_path:
                    for _, name, is_pkg in pkgutil.iter_modules([pkg_path]):
                        if not name.startswith('_'):
                            try:
                                importlib.import_module(f'{pkg_name}.{name}')
                                logger.debug(f"Loaded module: {pkg_name}.{name}")
                            except Exception as e:
                                logger.warning(f"Failed to load {pkg_name}.{name}: {e}")
            except ImportError as e:
                logger.warning(f"Failed to import package {pkg_name}: {e}")
    
    def run(self):
        try:
            logger.info("Starting dialogue execution")
            session = self.registry.get("session")
            
            logger.debug(f"Total conversations: {session.num_conversationts}, Worker threads: {session.max_workers}")
            
            results = session.run_all_conversations()
            
            logger.info(f"Completed: {results['completed']}, Failed: {results['failed']}, Duration: {results['duration_seconds']:.2f}s")
            
            logger.info("Execution complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="User-Agent Customer Service Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config" / "default.yaml"),
        help="Config file path (default: config/default.yaml)"
    )
    
    # Use parse_known_args to capture all unknown arguments
    args, unknown = parser.parse_known_args()
    
    # Pass unknown arguments as CLI parameters
    cli_args = unknown
    
    app = Application(args.config, cli_args=cli_args if cli_args else None)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if app.log_to_file:
        log_file = os.path.join(app.logger_dir, "application.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, app.log_level),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger.info("User-Agent Customer Service Simulation Framework")
    logger.info(f"📝 Config File: {args.config}")
    logger.info(f"📄 Logger Dir: {app.logger_dir}")
    logger.debug(f"📊 Log Level: {app.log_level}")
    
    try:
        app.initialize()
        results = app.run()
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Application execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
