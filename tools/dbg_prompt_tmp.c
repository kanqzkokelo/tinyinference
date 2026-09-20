#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
int main(int argc, char**argv){
 setenv("TT_NO_GRAPH","1",1);
 const char* mp = argc>=2?argv[1]:"data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
 const char* pr = argc>=3?argv[2]:"Hello world test. Hello world test. Hello world test.";
 GGUFModel*m=gguf_load(mp); if(!m){printf("load fail\n");return 1;}
 BPETokenizer*tok=bpe_tokenizer_init(m); if(!tok){printf("bpe fail\n");return 1;}
 TTConfig cfg=tt_config_from_gguf(m,1024);
 GGUFTensor*t=gguf_get_tensor(m,"token_embd.weight"); cfg.vocab=(int)t->shape[t->ndim-1];
 printf("vocab=%d dim=%d\n",cfg.vocab,cfg.dim);
 int pt[2048]; int n=bpe_encode(tok,pr,pt,2048);
 printf("n_prompt=%d\n",n);
 for(int i=0;i<n;i++) printf(" pt[%d]=%d%s\n",i,pt[i],pt[i]>=cfg.vocab?" OOB!":"");
 Qwen2Engine*e=qwen2_engine_create(&cfg,m); if(!e){printf("create fail\n");return 1;}
 printf("engine ok, prefill %d...\n",n-1);
 cudaError_t ce=cudaGetLastError(); printf("pre-clear: %d %s\n",(int)ce,cudaGetErrorString(ce));
 int rc=qwen2_engine_prefill(e,pt,n-1);
 printf("prefill rc=%d pos=%d\n",rc,qwen2_engine_pos(e));
 ce=cudaGetLastError(); printf("post-prefill err: %d %s\n",(int)ce,cudaGetErrorString(ce));
 if(!rc){ float*h=(float*)malloc((size_t)cfg.vocab*4); int r2=qwen2_engine_step_logits(e,pt[n-1],h); printf("step H rc=%d\n",r2); free(h);}
 // second engine
 printf("creating second engine...\n");
 Qwen2Engine*e2=qwen2_engine_create(&cfg,m); if(!e2){printf("create2 fail\n");return 1;}
 printf("engine2 ok, prefill %d...\n",n);
 rc=qwen2_engine_prefill(e2,pt,n);
 printf("prefill2 rc=%d pos=%d\n",rc,qwen2_engine_pos(e2));
 ce=cudaGetLastError(); printf("post-prefill2 err: %d %s\n",(int)ce,cudaGetErrorString(ce));
 return 0;
}
