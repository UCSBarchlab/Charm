import base64
import logging
from collections import namedtuple
from traceback import format_exc

from ipykernel.kernelbase import Kernel

from .interpreter.parser import Program

args = namedtuple('arg', ['verbose', 'z3core', 'draw', 'mcsamples'])
args.verbose = False
# Bug found: when executing:
# Traceback (most recent call last):
#   File "/home/bill/Documents/research/Charm/venv/lib/python3.6/site-packages/Charm/charm_jupyter_kernel.py", line 48, in do_execute
#     result = Program(self.code_cache, args).run()
#   File "/home/bill/Documents/research/Charm/venv/lib/python3.6/site-packages/Charm/interpreter/parser.py", line 244, in run
#     result = interp.run()
#   File "/home/bill/Documents/research/Charm/venv/lib/python3.6/site-packages/Charm/interpreter/interpreter.py", line 1151, in run
#     consistent_and_determined, _ = self.convert_to_functional_graph_using_z3()
#   File "/home/bill/Documents/research/Charm/venv/lib/python3.6/site-packages/Charm/interpreter/interpreter.py", line 289, in convert_to_functional_graph_using_z3
#     solution, time = graph_transform_z3(input_map, input_eq_map)
#   File "/home/bill/Documents/research/Charm/venv/lib/python3.6/site-packages/Charm/interpreter/z3core.py", line 22, in graph_transform_z3
#     exec("%s = z3.Bool('%s')" % (inp, inp))
#   File "<string>", line 1, in <module>
# AttributeError: module 'z3' has no attribute 'Bool'
# TODO Fix it before setting z3core to True
args.z3core = False
args.draw = False
args.mcsamples = 100
kernel = True


# All modules incorporating with this kernel are expected to use the logging mechanism rather than print
# to generate output.
class CharmKernel(Kernel):
    implementation = "Charm"
    implementation_version = "v1.0"
    language = "Charm"
    language_version = "1.0"
    language_info = {
        'name': 'Charm',
        'mimetype': 'text/plain',
        'file_extension': '.charm'
    }
    banner = 'A Charm kernel'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_cache = ""

    def do_execute(self, code, silent, store_history=True, user_expressions=None, allow_stdin=False):
        self.send_response(
            stream=self.iopub_socket,
            msg_or_type='status',
            content={
                'execution_state': 'busy'
            }
        )
        self.code_cache += "\n" + code
        if self.__code_complete():
            try:
                logging.log(logging.DEBUG, self.code_cache)
                result = Program(self.code_cache, args).run()
                if not silent:
                    if 'raw' in result:
                        self.send_response(
                            stream=self.iopub_socket,
                            msg_or_type='display_data',
                            content={
                                'data': {
                                    'text/plain': str(dict(result['raw']))
                                }
                            }
                        )
                    if 'img' in result:
                        for image in result['img']:
                            if image is not None:
                                    self.send_response(
                                        stream=self.iopub_socket,
                                        msg_or_type='display_data',
                                        content={
                                            'data':{
                                                'image/png': base64.encodebytes(image).decode()
                                            }
                                        }
                                    )
                return {
                    "status": "ok",
                    "execution_count": self.execution_count
                }
            except Exception:
                # TODO more detailed error message
                self.send_response(
                    stream=self.iopub_socket,
                    msg_or_type='stream',
                    content={
                        'name': 'stderr',
                        'text': format_exc()
                    }
                )
                return {
                    "status": "error",
                    "execution_count": self.execution_count
                }
            finally:
                self.send_response(
                    stream=self.iopub_socket,
                    msg_or_type='status',
                    content={
                        'execution_state': 'idle'
                    }
                )
                self.code_cache = ''
        else:
            self.send_response(
                stream=self.iopub_socket,
                msg_or_type='status',
                content={
                    'execution_state': 'idle'
                }
            )
            return {
                "status": "ok",
                "execution_count": self.execution_count
            }

    def do_apply(self, content, bufs, msg_id, reply_metadata):
        pass

    def do_clear(self):
        pass

    def __code_complete(self):
        # TODO easy to hack, fix it
        return not self.code_cache.find('explore') == -1


if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=CharmKernel)
