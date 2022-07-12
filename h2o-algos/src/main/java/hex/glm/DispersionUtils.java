package hex.glm;

import Jama.LUDecomposition;
import Jama.Matrix;
import hex.DataInfo;
import water.DKV;
import water.Job;
import water.fvec.Frame;
import water.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DispersionUtils {
    /***
     * Estimate dispersion factor using maximum likelihood.  I followed section IV of the doc in 
     * https://h2oai.atlassian.net/browse/PUBDEV-8683 . 
     */
    public static double estimateGammaMLSE(GLMTask.ComputeGammaMLSETsk mlCT, double seOld, double[] beta, 
                                           GLMModel.GLMParameters parms, ComputationState state, Job job, GLMModel model) {
        double constantValue = mlCT._wsum + mlCT._sumlnyiOui - mlCT._sumyiOverui;
        DataInfo dinfo = state.activeData();
        Frame adaptedF = dinfo._adaptedFrame;
        long currTime = System.currentTimeMillis();
        long modelBuiltTime = currTime - model._output._start_time;
        long timeLeft = parms._max_runtime_secs > 0 ? (long) (parms._max_runtime_secs * 1000 - modelBuiltTime) : Long.MAX_VALUE;

        // stopping condition for while loop are:
        // 1. magnitude of iterative change to se < EPS
        // 2. there are more than MAXITERATIONS of updates
        // 2. for every 100th iteration, we check for additional stopping condition:
        //    a.  User requests stop via stop_requested;
        //    b.  User sets max_runtime_sec and that time has been exceeded.
        for (int index=0; index<parms._max_iterations_dispersion; index++) {
            GLMTask.ComputeDiTriGammaTsk ditrigammatsk = new GLMTask.ComputeDiTriGammaTsk(null, dinfo, job._key, beta,
                    parms, seOld).doAll(adaptedF);
            double numerator = mlCT._wsum*Math.log(seOld)-ditrigammatsk._sumDigamma+constantValue; // equation 2 of doc
            double denominator = mlCT._wsum/seOld - ditrigammatsk._sumTrigamma;  // equation 3 of doc
            double change = numerator/denominator;
            if (denominator == 0 || !Double.isFinite(change))
                return seOld;
            if (Math.abs(change) < parms._dispersion_epsilon) // stop if magnitude of iterative updates to se < EPS
                return seOld-change;
            else {
                double se = seOld - change;
                if (se < 0) // heuristic to prevent seInit <= 0
                    seOld *= 0.5;
                else
                    seOld = se;
            }

            if ((index % 100 == 0) && // check for additional stopping conditions for every 100th iterative steps
                    (job.stop_requested() ||  // user requested stop via stop_requested()
                            (System.currentTimeMillis()-currTime) > timeLeft)) { // time taken to find dispersino exceeds GLM model building time
                Log.warn("gamma dispersion parameter estimation was interrupted by user or due to time out.  Estimation " +
                        "process has not converged. Increase your max_runtime_secs if you have set maximum runtime for your " +
                        "model building process.");
                return seOld;
            }
        }
        Log.warn("gamma dispersion parameter estimation fails to converge within "+
                parms._max_iterations_dispersion+" iterations.  Increase max_iterations_dispersion or decrease " +
                "dispersion_epsilon.");
        return seOld;
    }

    /**
     * This method estimates the tweedie dispersion parameter when variance power > 2.  It will use Newton's update
     * when it is working correctly:  meaning the log likelihood increase with the new dispersion parameter.  However,
     * there are cases where the log likelihood decreases with the new dispersion parameter, in this case, instead
     * of 
     */
    public static double estimateTweedieDispersionOnlyExceed2(GLMModel.GLMParameters parms, GLMModel model, Job job,
                                                       double[] beta, DataInfo dinfo) {
        long currTime = System.currentTimeMillis();
        long modelBuiltTime = currTime - model._output._start_time;
        long timeLeft = parms._max_runtime_secs > 0 ? (long) (parms._max_runtime_secs * 1000 - modelBuiltTime)
                : Long.MAX_VALUE;

        TweedieMLDispersionOnly tDispersion = new TweedieMLDispersionOnly(parms.train(), parms, model, beta, dinfo);
        double seCurr = tDispersion._dispersionParameter;   // initial value of dispersion parameter
        double seNew;
        double change, se, numerator, denominator;
        double logLLCurr;
        double logLLNext = Double.MIN_VALUE;
        List<Double> logValues = new ArrayList<>();
        List<Double> seValues = new ArrayList<>();
        List<Double> logDiff = new ArrayList<>();

        for (int index = 0; index < parms._max_iterations_dispersion; index++) {
            tDispersion.updateDispersionP(seCurr);
            DispersonTask.ComputeMaxSumSeriesTsk computeTask = new DispersonTask.ComputeMaxSumSeriesTsk(tDispersion,
                    parms);
            computeTask.doAll(tDispersion._infoFrame);
            logLLCurr = computeTask._logLL / computeTask._nobsLL;
            //logLLCurr = computeTask._logLL;
            
            // record log values
            logValues.add(logLLCurr);
            seValues.add(seCurr);
            if (logValues.size() > 1) {
                logDiff.add(logValues.get(index) - logValues.get(index - 1));
                if (Math.abs(logDiff.get(logDiff.size() - 1)) < parms._dispersion_epsilon) {
                    tDispersion.cleanUp();
                    return seValues.get(index);
                }
            }
            // set new alpha
            numerator = computeTask._dLogLL;
            denominator = computeTask._d2LogLL;
            change = numerator / denominator; // no line search is employed at the moment ToDo: add line search
            if (denominator == 0 || !Double.isFinite(change) || !Double.isFinite(logLLCurr))
                return seCurr;

            seNew = seCurr - change;
            tDispersion.updateDispersionP(seNew);
            DispersonTask.ComputeMaxSumSeriesTsk computeTaskNew = new DispersonTask.ComputeMaxSumSeriesTsk(tDispersion,
                    parms);
            computeTaskNew.doAll(tDispersion._infoFrame);
            logLLNext = computeTaskNew._logLL / computeTaskNew._nobsLL;
            //logLLNext = computeTaskNew._logLL;

            if (logLLNext > logLLCurr) { // there is improvement
                se = seNew;
            } else {    // logLL is better before, direction of change is wrong, use interpolation to get se
                if (logLLNext <= logLLCurr && logValues.size() >= parms._logll_length) {
                    return seValues.get(logValues.indexOf(Collections.max(logValues)));
                }
                se = seCurr + parms._dispersion_learning_rate * change;
            }

            if (se < 0)
                seCurr *= 0.5;
            else
                seCurr = se;

            if ((index % 100 == 0) && // check for additional stopping conditions for every 100th iterative steps
                    (job.stop_requested() ||  // user requested stop via stop_requested()
                            (System.currentTimeMillis() - currTime) > timeLeft)) { // time taken exceeds model build time
                Log.warn("tweedie dispersion parameter estimation was interrupted by user or due to time out." +
                        "  Estimation process has not converged. Increase your max_runtime_secs if you have set " +
                        "maximum runtime for your model building process.");
                tDispersion.cleanUp();
                return seCurr;
            }
        }
        tDispersion.cleanUp();
        return seCurr;
    }
    
    public static double[] makeZeros(double[] sourceCoeffs, double[] targetCoeffs) {
        int size = targetCoeffs.length;
        for (int valInd = 0; valInd < size; valInd++)
            targetCoeffs[valInd] = targetCoeffs[valInd]-sourceCoeffs[valInd];
        return targetCoeffs;
    }
    

    public static double estimateTweedieDispersionOnly(GLMModel.GLMParameters parms, GLMModel model, Job job, 
                                                       double[] beta, DataInfo dinfo) {
        if (parms._tweedie_variance_power > 2)
            return estimateTweedieDispersionOnlyExceed2(parms, model, job, beta, dinfo);

        long currTime = System.currentTimeMillis();
        long modelBuiltTime = currTime - model._output._start_time;
        long timeLeft = parms._max_runtime_secs > 0 ? (long) (parms._max_runtime_secs * 1000 - modelBuiltTime) 
                : Long.MAX_VALUE;

        TweedieMLDispersionOnly tDispersion = new TweedieMLDispersionOnly(parms.train(), parms, model, beta, dinfo);
        double seOld = tDispersion._dispersionParameter;   // initial value of dispersion parameter
        double change, se, numerator, denominator;
        List<Double> logValues = new ArrayList<>();
        List<Double> seValues = new ArrayList<>();
        List<Double> logDiff = new ArrayList<>();
        
        for (int index=0; index<parms._max_iterations_dispersion; index++) {
            DispersonTask.ComputeMaxSumSeriesTsk computeTask = new DispersonTask.ComputeMaxSumSeriesTsk(tDispersion,
                    parms);
            computeTask.doAll(tDispersion._infoFrame);
            //DKV.put(tDispersion._infoFrame);
            if (parms._debugTDispersionOnly) {
                logValues.add(computeTask._logLL);
                seValues.add(seOld);
                if (logValues.size() > 1)
                    logDiff.add(logValues.get(index)-logValues.get(index-1));
            }
            // set new alpha
            numerator = computeTask._dLogLL;
            denominator = computeTask._d2LogLL;
            change = numerator/denominator; // no line search is employed at the moment ToDo: add line search
            if (denominator == 0 || !Double.isFinite(change)) {
                tDispersion.cleanUp();
                return seOld;
            }
            if (Math.abs(change) < parms._dispersion_epsilon) {
                tDispersion.cleanUp();
                return seOld - change;
            } else {
                se = seOld - change;
                if (se < 0)
                    seOld *= 0.5;
                else
                    seOld = se;
            }
            tDispersion.updateDispersionP(seOld);
            // set step size ??
            if ((index % 100 == 0) && // check for additional stopping conditions for every 100th iterative steps
                    (job.stop_requested() ||  // user requested stop via stop_requested()
                            (System.currentTimeMillis()-currTime) > timeLeft)) { // time taken exceeds model build time
                Log.warn("tweedie dispersion parameter estimation was interrupted by user or due to time out." +
                        "  Estimation process has not converged. Increase your max_runtime_secs if you have set " +
                        "maximum runtime for your model building process.");
                tDispersion.cleanUp();
                return seOld;
            }
        }
        tDispersion.cleanUp();
        return seOld;
    }
}
