import ipdb
import subprocess
import time

DEF_INSTANCE = 'c5.large'

def args_to_dict(args):
	if isinstance(args,str):
		args = args.split(' ')
	rc = {}
	for k in args:
		a, b = k.split('=')
		rc[a]=b		
	return rc

def dict_to_args(d, as_string=True):
	rc = []
	for (a,b) in d.items():
		rc.append(a+'='+str(b))
	if as_string:
		rc = " ".join(rc)
	return rc

def get_mongo_addr(machine):
	rc = subprocess.run(['docker-machine', 'ip', machine], stdout=subprocess.PIPE)
	assert(rc.returncode == 0)
	ip = rc.stdout.strip().decode()

	return ip

def machine_exists(name):
	bla = subprocess.run(['docker-machine', 'inspect', name], stdout=subprocess.PIPE)
	return bla.returncode == 0

def machine_name(name):
	sanitized_name = ['-' if x == '_' else x for x in name]
	return ''.join(sanitized_name)+'-'+str(time.time())[-4:]

def provision_machine(name, instance_type=None):
	if not instance_type:
		instance_type = DEF_INSTANCE
	if not machine_exists(name):
		rc = subprocess.run(['./provision.sh', name, instance_type])
	else:
		print('Machine %s already exists!')	
	assert(machine_exists(name))

def remove_machine(name):
	rc = subprocess.run(['docker-machine', 'rm', '-y', name])
	assert(not machine_exists(name))

def execute_machine(name, args):
	assert(machine_exists(name))
	rc = subprocess.run(['./start-container.sh', name, args])
	
