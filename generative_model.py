import json
import os, sys
import numpy as np
import tensorflow as tf
import random

import model, sample, encoder

from googletrans import Translator

class GenerativeModel():
    def __init__(
        self,
        model_name='124M',
        seed=None,
        length=None,
        temperature=0.8,
        top_k=0,
        top_p=1,
        models_dir='./models',
        verbose=True):

        self.model_name  = model_name
        self.seed        = seed
        self.batch_size  = 1
        self.length      = length
        self.temperature = temperature
        self.top_k       = top_k
        self.top_p       = top_p
        self.models_dir  = models_dir
        self.verbose     = verbose

        
        if self.batch_size is None:
            self.batch_size = 1
            
        
        self._load_model()

        self.translator = None
        return None


    def _load_model(self):
        
        if self.verbose:
            print(' - Loading model: ...')
            
        self.enc     = encoder.get_encoder(self.model_name, self.models_dir)
        self.hparams = model.default_hparams()
        
        with open(os.path.join(self.models_dir, self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
            if self.verbose:
                print(' - Setting: length to:', self.length)
            
        elif self.length > self.hparams.n_ctx:
            self.length = self.hparams.n_ctx // 2
            print("Can't get samples longer than window size: %s, using default value." % hparams.n_ctx, file=sys.stderr)



        self.sess = tf.compat.v1.Session(graph=tf.get_default_graph())

        self.context = tf.compat.v1.placeholder(tf.int32, [self.batch_size, None])
        
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        
        self.output = sample.sample_sequence(
            hparams=self.hparams,
            length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
        )
        
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.latest_checkpoint( os.path.join(self.models_dir, self.model_name) )
        self.saver.restore(self.sess, self.ckpt)

        if self.verbose:
            print(' - Loading model: OK!!')
            
        return None


    def translate(self, text, input_lang='en', output_lang='es'):
        if self.translator is None:
            translator = Translator()

        tr_obj = translator.translate(text, src=input_lang, dest=output_lang)
        return tr_obj.text
    
            
    def gen_from_sample(self, raw_text='Mi querido hijo, no sabes la alegría que me dió leer tu carta', nsamples=1, input_lang='es', output_lang='es'):

        if input_lang != 'en':
            raw_text = self.translate(raw_text,
                                      input_lang=input_lang,
                                      output_lang='en')
            
        context_tokens = self.enc.encode(raw_text)
        generated_text_v = [raw_text]
        generated = 0


        assert nsamples % self.batch_size == 0
            
        for _ in range(nsamples // self.batch_size):
            out = self.sess.run(self.output, feed_dict={self.context: [context_tokens for _ in range(self.batch_size)]})[:, len(context_tokens):]

            for i in range(self.batch_size):
                if self.verbose:
                    print(' Generating Sample:', generated, '...')
                    
                generated += 1
                text = self.enc.decode(out[i])
                generated_text_v.append(raw_text + ' ' + text)


        if output_lang != 'en':
            for i in range(len(generated_text_v)):
                generated_text_v[i] = self.translate(generated_text_v[i],
                                                     input_lang='en',
                                                     output_lang=output_lang)

##        if self.verbose:
##            print("=" * 80)
##            print(' Input text:', raw_text)
##            print("=" * 80)
##            print()
##
##            for i in range(len(generated_text_v)):
##                print("=" * 40 + " SAMPLE " + str(i-1) + " " + "=" * 40)
##                print(generated_text_v[i])
##
##            print("=" * 80)
            
        return generated_text_v


if __name__ == '__main__':
    gen_model = GenerativeModel(length=256)
    generated_text_v = gen_model.gen_from_sample(nsamples=3)


    for i in range(len(generated_text_v)):
        print("=" * 40 + " SAMPLE " + str(i-1) + " " + "=" * 40)
        print(generated_text_v[i])

    print("=" * 80)
