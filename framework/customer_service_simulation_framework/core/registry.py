from typing import Dict, Type, Any, Optional, List
import logging
import os
import yaml
import json

logger = logging.getLogger(__name__)


def convert_value(value: str):
    """
    Convert a string value to the appropriate type

    Args:
        value: String value

    Returns:
        Converted value
    """
    # Handle JSON objects (e.g., dictionaries)
    if value.startswith('{') and value.endswith('}'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    # Handle JSON arrays
    if value.startswith('[') and value.endswith(']'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    # Try to convert to integer
    if value.isdigit():
        return int(value)
    
    # Try to convert to float
    try:
        if '.' in value and value.replace('.', '', 1).replace('-', '', 1).isdigit():
            return float(value)
    except (ValueError, AttributeError):
        pass
    
    # Try to convert to boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Return as string
    return value


def apply_overrides(config: dict, overrides: dict) -> dict:
    """
    Apply override parameters to the configuration

    Supports finding components by name:
    - agent.customer.model -> find the item with name=customer in the agent list, modify its model
    - session.RLSimulation.output_dir -> find the item with name=RLSimulation in the session list, modify its output_dir

    Args:
        config: Original configuration dictionary
        overrides: Override parameters dictionary

    Returns:
        Updated configuration dictionary
    """
    def apply_override_recursive(base, override, path=[]):
        """Recursively apply override parameters"""
        for key, value in override.items():
            current_path = path + [key]
            
            if isinstance(value, dict):
                # Check if this is a component type (agent or session)
                if key in ('agent', 'session') and isinstance(base.get(key), list):
                    # This is a component list, keys in value are component names
                    for component_name, component_overrides in value.items():
                        # Find the component with matching name in the list
                        for component in base[key]:
                            if isinstance(component, dict) and component.get('name') == component_name:
                                # Found matching component, apply overrides
                                apply_override_recursive(component, component_overrides, current_path + [component_name])
                                break
                else:
                    # Regular nested dictionary
                    if key not in base:
                        base[key] = {}
                    if isinstance(base[key], dict):
                        apply_override_recursive(base[key], value, current_path)
            else:
                # Set value directly
                base[key] = value
    
    apply_override_recursive(config, overrides)
    return config


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        # Prefer using the provided config dictionary
        if config_dict is not None:
            self.config = config_dict
            logger.debug("Using provided config dictionary")
            return
        
        if config_path is None:
            raise Exception("No config file path or config dictionary specified")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.debug(f"Loaded config file: {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Config file YAML format error: {e}")
        except Exception as e:
            raise Exception(f"Failed to load config file: {e}")
    
    def get_component_config(self, component_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Get component configuration by type and name

        Args:
            component_type: Component type, e.g., 'agent', 'session'
            name: Component name, e.g., 'user', 'customer'
        """
        if component_type not in self.config:
            raise Exception(f"Component type not found in config: {component_type}")
        
        component_configs = self.config[component_type]
        
        # If it's a list, iterate to find the matching name
        if isinstance(component_configs, list):
            for config in component_configs:
                if isinstance(config, dict) and config.get('name') == name:
                    return config
            raise Exception(f"Component {name} of type {component_type} not found in config")

        # If it's a dict, return it directly (single component case)
        elif isinstance(component_configs, dict):
            if component_configs.get('name') == name:
                return component_configs
            raise Exception(f"Config {component_type} name mismatch: expected {name}, got {component_configs.get('name')}")

        raise Exception(f"Config format error: {component_type} should be a dict or list")


class ComponentRegistry:
    # Fixed component types
    COMPONENT_TYPES = {'agent', 'session'}
    
    def __init__(self, config_loader: ConfigLoader) -> None:
        # _registries: {component_type: {name: component_class}}
        # e.g.: {'agent': {'user': UserAgentClass, 'customer': CustomerAgentClass}}
        self._registries: Dict[str, Dict[str, Type]] = {ct: {} for ct in self.COMPONENT_TYPES}
        
        # _instances: {component_type: {name: instance}}
        # e.g.: {'agent': {'user': user_agent_instance, 'customer': customer_agent_instance}}
        self._instances: Dict[str, Dict[str, Any]] = {ct: {} for ct in self.COMPONENT_TYPES}
        
        self._config_loader = config_loader
        self._config_names = self._discover_config_names()

    def _discover_config_names(self) -> Dict[str, List[str]]:
        """Discover all component names from the config file"""
        if not self._config_loader or not self._config_loader.config:
            logger.warning("Config loader not set or config is empty")
            return {ct: [] for ct in self.COMPONENT_TYPES}
        
        config_names = {ct: [] for ct in self.COMPONENT_TYPES}
        
        for component_type in self.COMPONENT_TYPES:
            if component_type not in self._config_loader.config:
                continue
            
            component_configs = self._config_loader.config[component_type]
            
            # If it's a list, extract all names
            if isinstance(component_configs, list):
                for config in component_configs:
                    if isinstance(config, dict) and 'name' in config:
                        name = config['name']
                        config_names[component_type].append(name)
                        logger.debug(f"Found {component_type} config: {name}")

            # If it's a dict, extract the single name
            elif isinstance(component_configs, dict) and 'name' in component_configs:
                name = component_configs['name']
                config_names[component_type].append(name)
                logger.debug(f"Found {component_type} config: {name}")
        
        return config_names
    
    def get_config(self, component_type: str, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration by component type and name"""
        if not self._config_loader:
            logger.warning("Config loader not set")
            return None
        
        try:
            return self._config_loader.get_component_config(component_type, name)
        except Exception as e:
            logger.error(f"Failed to get component config: {e}")
            return None

    def register(self, component_type: str, name: str, component_class: Type) -> None:
        """Register a component class

        Args:
            component_type: Component type, must be 'agent' or 'session'
            name: Component name, e.g., 'user', 'customer'
            component_class: Component class
        """
        if component_type not in self.COMPONENT_TYPES:
            raise Exception(f"Unknown component type: {component_type}, must be one of {self.COMPONENT_TYPES}")

        if name in self._registries[component_type]:
            logger.warning(f"Component {name} (type: {component_type}) is already registered")

        self._registries[component_type][name] = component_class
        logger.debug(f"Registered component: {component_type}.{name}")
    
    def init_component(self) -> None:
        """Initialize all registered components"""
        # Check that session has only one
        if 'session' in self._config_names and len(self._config_names['session']) > 1:
            raise Exception(f"Only one session can be configured, but found {len(self._config_names['session'])} in config: {self._config_names['session']}")

        # Initialize components in fixed order: agents first, then sessions
        # This ensures all dependent agents are initialized before sessions
        component_types_ordered = ['agent', 'session']
        
        for component_type in component_types_ordered:
            if component_type not in self.COMPONENT_TYPES:
                continue
            for name in self._config_names.get(component_type, []):
                if name in self._registries[component_type]:
                    component_class = self._registries[component_type][name]
                    self._auto_init_component(component_type, name, component_class)
                else:
                    logger.warning(f"Component {component_type}.{name} in config is not registered")

    def _auto_init_component(self, component_type: str, name: str, component_class: Type) -> bool:
        """Automatically initialize a single component"""
        try:
            config = self.get_config(component_type, name)
            if config is None:
                logger.warning(f"Unable to get config: {component_type}.{name}")
                return False
            
            # Create component instance
            instance = component_class(component_type=component_type, name=name, config=config)
            self._instances[component_type][name] = instance
            logger.debug(f"Component initialized successfully: {component_type}.{name}")
            return True
        except Exception as e:
            logger.error(f"Component initialization failed: {component_type}.{name}, error: {e}")
            return False
    
    def get(self, component_type: str, name: Optional[str] = None) -> Any:
        """Get a component instance by type and name

        Args:
            component_type: Component type, e.g., 'agent', 'session'
            name: Component name, e.g., 'user', 'customer'.
                  For session type, if name is not specified, the single session instance is returned automatically

        Returns:
            Component instance
        """
        if component_type not in self.COMPONENT_TYPES:
            raise Exception(f"Unknown component type: {component_type}")

        # If it's a session and name is not specified, automatically get the single session
        if component_type == 'session' and name is None:
            instances = self._instances[component_type]
            if len(instances) == 0:
                raise Exception("Session component not initialized")
            elif len(instances) > 1:
                raise Exception(f"Multiple session instances exist: {list(instances.keys())}, please specify name")
            # Return the single session instance
            return next(iter(instances.values()))
        
        # For agent or when name is specified
        if name is None:
            raise Exception(f"Name must be specified when getting {component_type} component")

        if name not in self._instances[component_type]:
            raise Exception(f"Component not initialized: {component_type}.{name}")
        
        return self._instances[component_type][name]
    
    def get_all_instances(self, component_type: str) -> Dict[str, Any]:
        """Get all component instances of a given type

        Args:
            component_type: Component type, 'agent' or 'session'

        Returns:
            {name: instance} dictionary
        """
        if component_type not in self.COMPONENT_TYPES:
            raise Exception(f"Unknown component type: {component_type}")
        
        return self._instances[component_type].copy()
    

_registry: Optional[ComponentRegistry] = None


def init_registry(config_path: str, cli_args: Optional[List[str]] = None) -> ComponentRegistry:
    """
    Initialize the registry, supporting CLI argument overrides

    Args:
        config_path: Config file path
        cli_args: CLI argument list, format: ['key=value', ...]

    Returns:
        ComponentRegistry instance
    """
    global _registry
    if _registry is not None:
        logger.warning("Registry already initialized")
    
    # Load config file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded config file: {config_path}")

    # Apply CLI argument overrides
    if cli_args:
        overrides = parse_cli_args(cli_args)
        config = apply_overrides(config, overrides)
        logger.info(f"Applied CLI argument overrides")

    # Initialize ConfigLoader with config dictionary
    config_loader = ConfigLoader(config_dict=config)
    _registry = ComponentRegistry(config_loader)
    logger.debug(f"Registry initialized with config: {config_path}")
    return _registry


def parse_cli_args(cli_args: List[str]) -> dict:
    """
    Parse CLI arguments

    Supported formats:
    - agent.customer.llm.model=qwen-turbo
    - session.RLSimulation.output_dir=/path/to/output
    - agent.user.llm.kwargs.temperature=0.5

    Args:
        cli_args: CLI argument list, format: key=value

    Returns:
        Nested dictionary representing config overrides
    """
    overrides = {}
    
    if not cli_args:
        return overrides
    
    for arg in cli_args:
        if '=' not in arg:
            logger.warning(f"Skipping invalid argument: {arg}, should be key=value format")
            continue
        
        key_path, value = arg.split('=', 1)
        
        # Remove leading -- or -
        if key_path.startswith('--'):
            key_path = key_path[2:]
        elif key_path.startswith('-'):
            key_path = key_path[1:]
        
        # Try to convert value to the appropriate type
        converted_value = convert_value(value)
        
        # Build nested dictionary
        keys = key_path.split('.')
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the last key
        final_key = keys[-1]
        current[final_key] = converted_value
        
        logger.debug(f"Parameter: {key_path} = {converted_value}")
    
    return overrides


def get_registry() -> ComponentRegistry:
    if _registry is None:
        raise RuntimeError("Registry not initialized")
    return _registry


def register_component(component_type: str, name: str):
    """Decorator for registering components

    Args:
        component_type: Component type, 'agent' or 'session'
        name: Component name, e.g., 'user', 'customer'

    Example:
        @register_component(component_type='agent', name='user')
        class UserAgent:
            pass
    """
    def decorator(cls):
        registry = get_registry()
        registry.register(component_type, name, cls)
        return cls
    return decorator
