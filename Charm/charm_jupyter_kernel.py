import logging
from collections import namedtuple
from traceback import format_exc

from ipykernel.kernelbase import Kernel

from .interpreter.parser import Program

args = namedtuple('arg', ['verbose', 'z3core', 'draw', 'mcsamples'])
args.verbose = False
args.z3core = True
args.draw = False
args.mcsamples = 100
kernel = True
logging.basicConfig(level=logging.DEBUG)


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
                        for filename in result['img']:
                            if filename is not None:
                                with open(filename,'rb') as f:
                                    img=f.read()
                                    self.send_response(
                                        stream=self.iopub_socket,
                                        msg_or_type='display_data',
                                        content={
                                            'data':{
                                                'image/jpeg':img
                                            }
                                        }
                                    )
                return {
                    "status": "ok",
                    "execution_count": self.execution_count
                }
            except Exception as e:
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
