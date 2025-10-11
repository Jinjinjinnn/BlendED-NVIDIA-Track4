import json


class SimConfig:
    def __init__(self, scene_file_path=None, config_dict=None):
        if config_dict is not None:
            self.cfg = config_dict
        elif scene_file_path is not None:
            with open(scene_file_path, 'r') as f:
                self.cfg = json.load(f)
        else:
            # Create a default config if nothing is provided
            self.cfg = {
                "domainStart": [0.0, 0.0, 0.0],
                "domainEnd": [1.0, 1.0, 1.0],
                "simulationMethod": 4, # DFSPH
                "particleRadius": 0.01,
                "timeStepSize": 1e-4,
                "fluid_blocks": [],
                "rigid_blocks": [],
                "rigid_bodies": [],
            }

    def get_cfg(self, key):
        if key in self.cfg:
            return self.cfg[key]
        
        # Look for the key in sph_params if it exists
        if "sph_params" in self.cfg and key in self.cfg["sph_params"]:
            return self.cfg["sph_params"][key]

        # Fallback for nested keys in the old format for compatibility
        if key == "numberOfStepsPerRenderUpdate":
            return int(0.016 / self.get_cfg("timeStepSize"))
        
        print(f" WARN: {key} not found in config, returning None")
        return None

    def get_fluid_blocks(self):
        if "fluid_blocks" in self.cfg:
            return self.cfg["fluid_blocks"]
        return []

    def get_rigid_blocks(self):
        if "rigid_blocks" in self.cfg:
            return self.cfg["rigid_blocks"]
        return []
    
    def get_rigid_bodies(self):
        # Adapt to the new boundary_conditions format
        if "boundary_conditions" in self.cfg:
            rigid_bodies = []
            for bc in self.cfg["boundary_conditions"]:
                if bc.get("type") == "rigid_body":
                    rigid_bodies.append(bc)
            return rigid_bodies
        # Fallback to old format
        if "rigid_bodies" in self.cfg:
            return self.cfg["rigid_bodies"]
        return []

    def get_boundary_conditions(self):
        if "boundary_conditions" in self.cfg:
            return self.cfg["boundary_conditions"]
        return []
