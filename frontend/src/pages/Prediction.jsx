import React, { useState, useEffect } from 'react';
import { Formik, Form, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import { fighterService, predictionService } from '../services/api';
import PredictionResult from '../components/prediction/PredictionResult';
import FighterSelector from '../components/fighters/FighterSelector';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

const validationSchema = Yup.object({
  fighter1: Yup.string().required('First fighter is required'),
  fighter2: Yup.string().required('Second fighter is required')
    .test('different-fighters', 'Select different fighters', function(value) {
      return value !== this.parent.fighter1;
    })
});

const Prediction = () => {
  const [fighters, setFighters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    const loadFighters = async () => {
      try {
        setLoading(true);
        const data = await fighterService.getAllFighters();
        setFighters(data);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load fighters:", err);
        setError("Failed to load fighter data. Please try again later.");
        setLoading(false);
      }
    };

    loadFighters();
  }, []);

  const handleSubmit = async (values, { setSubmitting }) => {
    try {
      setPredicting(true);
      setPredictionResult(null);
      setError(null);
      
      // Get full fighter details
      const fighter1Data = await fighterService.getFighterByName(values.fighter1);
      const fighter2Data = await fighterService.getFighterByName(values.fighter2);
      
      // Make prediction
      const result = await predictionService.predictFight(fighter1Data, fighter2Data);
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction failed:", err);
      setError("Failed to get prediction. Please try again.");
    } finally {
      setPredicting(false);
      setSubmitting(false);
    }
  };

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Predict a UFC Fight</h1>
      
      {loading ? (
        <LoadingSpinner text="Loading fighter data..." />
      ) : error ? (
        <ErrorAlert message={error} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="card">
            <h2 className="text-2xl font-bold mb-4">Fighter Selection</h2>
            <Formik
              initialValues={{ fighter1: '', fighter2: '' }}
              validationSchema={validationSchema}
              onSubmit={handleSubmit}
            >
              {({ isSubmitting }) => (
                <Form>
                  <div className="mb-4">
                    <label className="block text-gray-300 mb-2">
                      Fighter 1 (Red Corner)
                    </label>
                    <FighterSelector
                      name="fighter1"
                      fighters={fighters}
                      cornerClass="text-ufc-red"
                    />
                    <ErrorMessage name="fighter1" component="div" className="text-ufc-red text-sm mt-1" />
                  </div>
                  
                  <div className="mb-6">
                    <label className="block text-gray-300 mb-2">
                      Fighter 2 (Blue Corner)
                    </label>
                    <FighterSelector
                      name="fighter2"
                      fighters={fighters}
                      cornerClass="text-ufc-blue"
                    />
                    <ErrorMessage name="fighter2" component="div" className="text-ufc-red text-sm mt-1" />
                  </div>
                  
                  <button
                    type="submit"
                    className="w-full btn-primary"
                    disabled={isSubmitting || predicting}
                  >
                    {predicting ? 'Analyzing Fight...' : 'Predict Fight'}
                  </button>
                </Form>
              )}
            </Formik>
          </div>
          
          <div>
            {predicting ? (
              <div className="card text-center py-12">
                <LoadingSpinner text="Analyzing fighters..." />
                <p className="text-gray-400 mt-4">Our AI is calculating the fight outcome</p>
              </div>
            ) : predictionResult ? (
              <PredictionResult result={predictionResult} />
            ) : (
              <div className="card text-center py-12">
                <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>
                <p className="text-gray-400">
                  Select two fighters and click "Predict Fight" to see the prediction.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      
      <div className="mt-12 bg-gray-900/50 card">
        <h3 className="text-xl font-bold mb-4">How Our Prediction Works</h3>
        <p className="text-gray-300 mb-3">
          Our AI model analyzes various factors to predict fight outcomes:
        </p>
        <ul className="list-disc pl-5 text-gray-300 space-y-2">
          <li>Fighter's past performance history</li>
          <li>Technical statistics (striking accuracy, takedown defense)</li>
          <li>Physical attributes (reach advantage, height, weight)</li>
          <li>Fighting style matchups</li>
          <li>Recent performance trends</li>
        </ul>
        <p className="text-gray-400 mt-4 text-sm">
          Disclaimer: Predictions are based on statistical models and historical data. 
          UFC fights are inherently unpredictable, and our model provides probabilities rather than guarantees.
        </p>
      </div>
    </div>
  );
};

export default Prediction;