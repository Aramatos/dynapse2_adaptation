import errno
import os.path
import subprocess
import re
import time

import samna
from samna.dynapse2 import *

def readable(file):
    try:
        f = open(file, mode='rb', buffering=0)
    except OSError:
        return False
    f.close()
    return True


def bus_and_dev_from_vid_pid_string(vid_pid_string):
    cp = subprocess.run(['lsusb', '-d', vid_pid_string], stdout=subprocess.PIPE,
                        universal_newlines=True)
    if cp.returncode != 0:
        print('Dynap-se2 not found')
        exit(errno.ENODEV)

    regexp = re.compile('Bus (\d{3}) Device (\d{3}): ID [0-9a-f]{4}:[0-9a-f]{4} *.')
    m = regexp.match(cp.stdout)
    if m is None:
        print('Unexpected output from lsusb: "' + cp.stdout + '"')
        exit(errno.EPROTO)

    return int(m.group(1)), int(m.group(2))


def connect(device, n_chips, samna_node, sender_endpoint, receiver_endpoint, node_id, interpreter_id):
    assert (node_id != interpreter_id)

    if device == 'devboard':
        vid_pid_string = '%04x:%04x' % samna_node.get_dynapse2_dev_board_vid_and_pid()
    if device == 'stack':
        vid_pid_string = '%04x:%04x' % samna_node.get_dynapse2_stack_vid_and_pid()
    bus, dev = bus_and_dev_from_vid_pid_string(vid_pid_string)
    print('Bus %03d Device %03d: ID %s' % (bus, dev, vid_pid_string))
    if device == 'devboard':
        samna_node.open_dynapse2_dev_board(bus, dev)
    if device == 'stack':
        samna_node.open_dynapse2_stack(bus, dev, n_chips)

    samna.setup_local_node(receiver_endpoint, sender_endpoint, interpreter_id)
    samna.open_remote_node(node_id, "device_node")

    return samna.device_node


def dynapse2board(opts, args, remote):
    if len(args) == 0:
        print('No bitfile specified')
        exit(errno.EINVAL)

    if len(args) > 2:
        print('Too many arguments')
        exit(errno.E2BIG)

    bitfile = args[0]

    if not os.path.isfile(bitfile):
        print('Bitfile %s not found' % bitfile)
        exit(errno.ENOENT)

    if not readable(bitfile):
        print('Cannot read %s' % bitfile)
        exit(errno.EACCES)

    if opts.device == 'devboard':
        board = remote.Dynapse2DevBoard
    if opts.device == 'stack':
        board = remote.Dynapse2Stack

    if not board.configure_opal_kelly(bitfile):
        print('Failed to configure Opal Kelly')
        exit(errno.EIO)

    time.sleep(0.1)
    board.reset_fpga()
    time.sleep(0.1)

    # time.sleep(1)

    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToWest,
    #                                       Dynapse2ModuleConfigName(0), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToWest,
    #                                       Dynapse2ModuleConfigName(1), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToWest,
    #                                       Dynapse2ModuleConfigName(2), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToII,
    #                                       Dynapse2ModuleConfigName(0), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToII,
    #                                       Dynapse2ModuleConfigName(1), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup.BusFpgaToII,
    #                                       Dynapse2ModuleConfigName(2), 10)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup(1),
    #                                       Dynapse2ModuleConfigName(4), 100)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup(1),
    #                                       Dynapse2ModuleConfigName(5), 1000)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup(1),
    #                                       Dynapse2ModuleConfigName(6), 1000)
    # board.set_fpga_dynapse2_module_config(Dynapse2ModuleConfigGroup(1),
    #                                       Dynapse2ModuleConfigName(7), 1000)

    # time.sleep(1)

    return board
