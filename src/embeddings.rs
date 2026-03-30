use crate::model::Layer;
use safetensors::SafeTensors;

struct RotaryEmbeddingLayer<'t> {
    xq: SafeTensors<'t>,
    xc: SafeTensors<'t>,
    freq_cis: SafeTensors<'t>,
}

impl<'p> Layer for RotaryEmbeddingLayer<'p> {
    fn new() -> Self where Self: Sized {
        unimplemented!()
    }

    fn forward<'t>(input: SafeTensors<'t>) -> SafeTensors<'t> {
        input
    }
}
