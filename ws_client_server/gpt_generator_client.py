import numpy as np
import os, sys
import pickle
import traceback
import time

from websocket_client import ws_client



def on_message(ws, message):
    pred_obj = ws.pred_obj

##    print('on_message:', message)
    try:
        if type(message) in [bytes, bytearray]:
            
            pred_d = pickle.loads(message)

            generation_dir = pred_obj.generation_dir
            gender_dir = pred_d['gender_folder']

            if not os.path.exists( generation_dir ):
                os.mkdir(generation_dir)

            if not os.path.exists( os.path.join(generation_dir, gender_dir) ):
                os.mkdir(os.path.join(generation_dir, gender_dir))

                
            i = 0
            if len(pred_d['generated_text_v']) > 1:
                for gen_text in pred_d['generated_text_v'][1:]:
                    not_saved = True
                    while not_saved:
                        save_path = os.path.join(generation_dir, gender_dir, '{:03d}.txt'.format(i))
                        
                        if not os.path.exists(save_path):
                            with open(save_path, 'w') as f:
                                f.write(gen_text)
                                
                            print(' Saved generated text:', save_path)
                            not_saved = False
                            
                        i += 1
                        if i > 1000:
                            raise Exception(' - ERROR, i > 1000.')
            else:
                raise Exception(' - ERROR, no generated text on message.')
                    
        else:
            print('on_message: type={} msg={}'.format(type(message), message))
            
    except Exception as e:
        print(' - ERROR on message analysis:', e)
        traceback.print_exc()
        
        
    return None


class GPTGeneatorClient():
    def __init__(self,
                 generation_dir='../gpt_generations',
                 host='localhost',
                 port=8001,
                 password='gpt_model'):
        
        self.generation_dir = generation_dir
        
        self.host = host
        self.port = port
        self.password = password


        self.client = ws_client(host=self.host,
                                port=self.port,
                                on_message_function=on_message,
                                password=self.password)

        return None



    def connect(self, timeout=10):
        if not self.client.connected:
            self.client.start()
            self.client.ws.pred_obj = self

            # Wait connection
            dt = 0.5
            t_w = 0
            t_s = time.time()
            while t_w < timeout:
                time.sleep(dt)
                t_w += dt
                if self.client.connected:
                    print(' Connected!!')
                    break
                else:
                    print('.', end='')
                
            if not self.client.connected:
                print(' - WARNING: the client is not connected to the server.')
                
        return None

    def close(self):
        if self.client.connected:
            self.client.close()
        
        return None

    def generate(self, text_seed='Mi querido hijo, no sabes la alegría que me dió leer tu carta', gender_folder='male', n_samples=1, input_lang='es', output_lang='es'):

        self.connect()
        
        if self.client.connected:
            data_d = {
                'raw_text':text_seed,
                'gender_folder':gender_folder,
                'nsamples':n_samples,
                'input_lang':input_lang,
                'output_lang':output_lang}

            data_bytes = pickle.dumps(data_d)
            self.client.send( data_bytes )

            print(' Message sent, wait for response ...')
        else:
            raise Exception(' - ERROR, the client is not connected ...')

            
        return None
        

    

            
if __name__ == '__main__':
    
    generator_client = GPTGeneatorClient()
    
    generator_client.generate(text_seed='Querida hijo como estas este día',
                              gender_folder='male',
                              n_samples=5,
                              )

    generator_client.generate(text_seed='Querida hija como estas este día',
                              gender_folder='female',
                              n_samples=5,
                              )



    
