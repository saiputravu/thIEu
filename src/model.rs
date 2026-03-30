use safetensors::SafeTensors;

pub(crate) trait Layer {
    fn new() -> Self where Self: Sized;
    // TODO: This should be our own tensor shape or some custom shape we can pass through.
    // Layer to layer, we expect the model to take different things.
    fn forward<'t>(input: SafeTensors<'t>) -> SafeTensors<'t>;
}
