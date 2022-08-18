package hex.pipeline;

import water.fvec.Frame;

public class UnionTransformer extends DataTransformer<UnionTransformer> {
  
  public enum UnionStrategy {
    add, 
    replace
  }
  
  private DataTransformer[] _transformers;
  private UnionStrategy _strategy;

  public UnionTransformer(DataTransformer[] transformers, UnionStrategy strategy) {
    _transformers = transformers;
    _strategy = strategy;
  }

  @Override
  public Frame transform(Frame fr, FrameType type, PipelineContext context) {
    Frame result = null;
    switch (_strategy) {
      case add:
        result = new Frame(fr);
        break;
      case replace:
        result = new Frame();
        break;
    }
    for (DataTransformer dt : _transformers) {
      result.add(dt.transform(fr, type, context));
    }
    return result;
  }
  
}
