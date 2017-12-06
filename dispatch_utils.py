import subprocess
import time

def get_mongo_addr(machine, sfx):
	rc = subprocess.run(['docker-machine', 'ip', machine], stdout=subprocess.PIPE)
	assert(rc.returncode == 0)
	ip = rc.stdout.strip().decode()

	return ip+sfx

def machine_exists(name):
	bla = subprocess.run(['docker-machine', 'inspect', name], stdout=subprocess.PIPE)
	return bla.returncode == 0

def machine_name(name):
  return str(name)+str(time.time())[-4:]

def provision_machine(name):
	rc = subprocess.run(['./provision.sh', name])
	assert(machine_exists(name))

def remove_machine(name):
	rc = subprocess.run(['docker-machine', '-y', 'rm', name])
	assert(not machine_exists(name))

def execute_machine(name, args):
	assert(machine_exists(name))
	rc = subprocess.run(['start-container', name, args])
	
