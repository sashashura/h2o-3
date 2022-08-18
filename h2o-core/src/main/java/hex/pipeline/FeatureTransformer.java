package hex.pipeline;

import water.Key;

/**
 * a DataTransformer that never modifies the response column
 */
public abstract class FeatureTransformer<T extends FeatureTransformer> extends DataTransformer<T> {
  
  private String[] _excluded_columns;

  public void excludeColumns(String[] columns) {
    _excluded_columns = columns;
  }
}
