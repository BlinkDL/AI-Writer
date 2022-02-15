########################################################################################################
# AI人工智障写作 - https://github.com/BlinkDL/AI-Writer
########################################################################################################

import math
import json
import random
import time

_DEBUG_LEVEL_ = 2  # 2 = full, 1 = partial, 0 = none
PORT_NUM = 8266

#
# 需要 pytorch 1.9.x 及以上版本
#
# gpu：只支持 nvidia 显卡，速度最快，需要 cuda+cudnn
# dml：支持 amd / intel / nvidia 显卡，需要不同的模型，需要 pip install onnxruntime-directml 然后在 run.py 和 server.py 设置为 dml 模式
# cpu：没显卡就选它，但也是用 nvidia 卡的模型

RUN_DEVICE = 'gpu' # gpu 或 dml 或 cpu

MODEL_NAME = 'model/wangwen-2022-02-15' # 模型名
WORD_NAME = 'model/wangwen-2022-02-15' # 这个也修改

top_p = 0.75 # 这个的范围是 0 到 1。越大，变化越多。越小，生成效果越规矩。自己试试 0 和 0.5 和 1.0 的效果就知道了
top_p_newline = 0.9

LENGTH_OF_EACH = 20  # 每次写多少字

ctx_len = 512
n_layer = 12
n_head = 12
n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

##############################################################################


def main():
    import sys
    import signal
    from multiprocessing import Process, RawArray, freeze_support, Queue, Lock

    freeze_support()

    queueZ = Queue()
    queueX = Queue()

    process = []
    process.append(Process(target=SocketWorker, args=(queueX, queueZ)))
    process.append(Process(target=NeuralWorker, args=(queueZ, queueX)))

    for p in process:
        p.daemon = True
        p.start()

    def signal_handler(signal, frame):
        for p in process:
            p.terminate()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for p in process:
        p.join()


def SocketWorker(queueX, queueZ):
    import asyncio
    import websockets
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    USERS = set()

    async def producer():
        hasData = False
        try:
            K, out = queueX.get(timeout=0.05)
            hasData = True
        except:
            pass
        if hasData:
            return (K, out)
        else:
            await asyncio.sleep(0.001)
            if random.random() < -0.003:
                return '[PING]'
            else:
                return ''

    async def producer_handler(websocket, path):
        while True:
            msg = await producer()
            if isinstance(msg, tuple):
                K, msg = msg
                for x in USERS:
                    if x.client_id == K:
                        # if _DEBUG_LEVEL_ > 0:
                        #     print('sent X', K)
                        await x.send(msg)
                        break
            elif msg != '':
                await websocket.send(msg)

    async def consumer(websocket, msg):
        if msg == '[PONG]':
            return
        try:
            msg = json.loads(msg)
            if msg['op'].lower() == 'get':
                # if _DEBUG_LEVEL_ > 0:
                #     print('get', websocket.client_id, msg['txt'])
                queueZ.put((websocket.client_id, msg['txt']))
        except Exception as e:
            print(e)
            pass

    async def consumer_handler(websocket, path):
        while True:
            msg = await websocket.recv()
            await consumer(websocket, msg)

    async def server(websocket, path):
        websocket.client_id = '%020x' % random.randrange(16**20)
        USERS.add(websocket)
        print("[ws connect]", len(USERS), 'users @',
              time.strftime("%Y %b %d %H:%M:%S", time.localtime(time.time())))
        try:
            await websocket.send('id_' + websocket.client_id)
            consumer_task = asyncio.ensure_future(
                consumer_handler(websocket, path))
            producer_task = asyncio.ensure_future(
                producer_handler(websocket, path))
            done, pending = await asyncio.wait(
                [consumer_task, producer_task],
                return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
        finally:
            USERS.remove(websocket)
            print("[ws disconnect]", len(USERS))

    def srv_exception(loop, context):
        if _DEBUG_LEVEL_ > 1:
            print('exception', loop, context)
        pass

    try:
        start_server = websockets.serve(server, "127.0.0.1", PORT_NUM)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().set_exception_handler(srv_exception)
        asyncio.get_event_loop().run_forever()
    except Exception as e:
        print('[srv error]', e)


def NeuralWorker(queueZ, queueX):
    from multiprocessing import Process, RawArray, freeze_support, Queue, Lock

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    import src.utils
    from src.model import GPT, GPTConfig

    # src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

    print('\nAI人工智障写作 https://github.com/BlinkDL/AI-Writer')
    print('请关注我的知乎 https://zhuanlan.zhihu.com/p/423646620')
    print('\n声明：模型的训练数据全部来自网文，缺乏生活常识。生成的文字仅供娱乐。请遵守法律法规。')

    print(f'\nLoading model for {RUN_DEVICE}...', end=' ')

    with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
        word_table = json.load(result_file)

    vocab_size = len(word_table)

    def train_dataset(): return None
    train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
    train_dataset.itos = {int(k): v for k, v in word_table.items()}
    UNKNOWN_CHAR = train_dataset.stoi['\ue083']

    if RUN_DEVICE == 'dml':
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = False
        rt_session = rt.InferenceSession(MODEL_NAME + '.onnx', sess_options=sess_options, providers=['DmlExecutionProvider'])
        rt_session.set_providers(['DmlExecutionProvider'])
    else:
        model = GPT(GPTConfig(vocab_size, ctx_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn))
        m2 = torch.load(MODEL_NAME + '.pth', map_location='cpu').state_dict()
        for i in range(n_layer):
            prefix = f'blocks.{i}.attn.'
            time_w = m2[prefix + 'time_w']
            time_alpha = m2[prefix + 'time_alpha']
            time_beta = m2[prefix + 'time_beta']
            
            TT = ctx_len
            T = ctx_len
            w = F.pad(time_w, (0, TT))
            w = torch.tile(w, [TT])
            w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
            w = w[:, :, TT-1:]
            w = w[:, :T, :T] * time_alpha[:, :, :T] * time_beta[:, :T, :]
            
            m2[prefix + 'time_ww'] = w
            del m2[prefix + 'time_w']
            del m2[prefix + 'time_alpha']
            del m2[prefix + 'time_beta']
        if RUN_DEVICE == 'gpu':
            model = model.cuda()
        model.load_state_dict(m2)

    print('done:', MODEL_NAME, '&', WORD_NAME)

    while True:
        K, Z = queueZ.get()
        # print('neural task', K, Z)

        ttt = time.time()

        context = Z
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        # print('您输入的开头有 ' + str(len(context)) +
        #       ' 个字。注意，模型只会看最后 ' + str(ctx_len) + ' 个字。')

        NUM_OF_RUNS = 1
        for run in range(NUM_OF_RUNS):

            x = np.array([train_dataset.stoi.get(s, UNKNOWN_CHAR)
                         for s in context], dtype=np.int64)

            real_len = len(x)
            print_begin = 0
            out_txt = ''

            for i in range(LENGTH_OF_EACH):

                if i == 0:
                    print_begin = real_len

                with torch.no_grad():
                    if RUN_DEVICE == 'dml':
                        if real_len < ctx_len:
                            xxx = np.pad(x, (0, ctx_len - real_len))
                        else:
                            xxx = x
                        out = rt_session.run(None, {rt_session.get_inputs()[0].name: [xxx[-ctx_len:]]})
                        out = torch.tensor(out[0])
                    else:
                        xxx = torch.tensor(x[-ctx_len:], dtype=torch.long)[None,...]
                        if RUN_DEVICE == 'gpu':
                            xxx = xxx.cuda()
                        out, _ = model(xxx)
                out[:, :, UNKNOWN_CHAR] = -float('Inf')

                pos = -1 if real_len >= ctx_len else real_len - 1

                if train_dataset.itos[int(x[real_len-1])] == '\n':
                    char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=top_p_newline)
                else:
                    char = src.utils.sample_logits(out, pos, temperature=1.0, top_p=top_p)

                x = np.append(x, char)
                real_len += 1

                completion = ''.join([train_dataset.itos[int(i)]
                                      for i in x[print_begin:real_len]])
                out_txt += completion
                print_begin = real_len

        outmsg = {}
        outmsg['op'] = 'TXT'
        outmsg['txt'] = out_txt
        queueX.put((K, json.dumps(outmsg, separators=(',', ':'))))

        # if _DEBUG_LEVEL_ > 1:
        #     print(time.time() - ttt, end=' ')
        ttt = time.time()
        if _DEBUG_LEVEL_ > 1:
            print(context, end = '')
            print(out_txt + '\n' + ('=' * 20))


if __name__ == "__main__":
    main()
