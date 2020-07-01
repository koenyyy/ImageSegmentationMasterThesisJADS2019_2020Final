import os
import json
from string import Template

class PipelineExecutor(object):
    """
    Object for doing feature extraction of a given dataset
    """
    def __init__(self):
        # Loading the json data from specified config file
        slurm_config_loc = os.path.join('config', 'slurm.json')
        with open(slurm_config_loc, 'r') as slurm_config:
            slurm_config_json = json.load(slurm_config)
        self.slurm_config_json = slurm_config_json
        self.platform = self.slurm_config_json['platform']

    def execute(self):
        print("EXECUTING PIPELINE:")

        print("   Creating slurm job:")
        self.create_slurmjob()
        # Create a slurmjob that calls the run all file

        # data_dir = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/data/input_data"
        # config_file = "C:/Users/s145576/Documents/GitHub/automaticSegmentationThesis/autoSeg/config/unet_end_to_end.json"
        # run_all.Run_all(data_dir, config_file).run()

    def create_slurmjob(self):
        print([type(self.slurm_config_json[i]) for i in self.slurm_config_json])
        if self.platform == 'cluster':
            cluster_slurm_template_loc = os.path.join('run', 'templates', 'cluster.job')
            with open(cluster_slurm_template_loc, 'r') as slurm_template:
                data = slurm_template.read()
                s = Template(data)
            #TODO zorgen dat dit goed word aangemaakt (out error copystring en execute)
            result = s.substitute(ntasks=self.slurm_config_json['ntasks'],
                                  mem=self.slurm_config_json['mem'],
                                  gres=self.slurm_config_json['gres'],
                                  timelimit=self.slurm_config_json['timelimit'],
                                  outfile=self.slurm_config_json['outfile'],
                                  errfile=self.slurm_config_json['errfile'],
                                  copystring= 'kak moet ik nog doen TODO',
                                  executestring='kak kak kak')

            # result = s.substitute({'ntasks': self.slurm_config_json.get('ntasks'),
            #                       'mem': self.slurm_config_json.get('mem'),
            #                       'gres': self.slurm_config_json.get('gres'),
            #                       'timelimit': self.slurm_config_json.get('timelimit'),
            #                       'outfile': self.slurm_config_json.get('outfile'),
            #                       'errfile': self.slurm_config_json.get('errfile')})

            slurm_bash_script_loc = os.path.join('run', 'slurm_bash.job')
            with open(slurm_bash_script_loc, 'w+') as outfile:
                outfile.write(result)

        elif self.platform != 'cluster':
            print('non cluster platforms have not yet been implemented')
            pass


if __name__ == '__main__':
    p = PipelineExecutor().execute()