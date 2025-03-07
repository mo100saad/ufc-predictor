import React, { useState, useEffect } from 'react';
import { Formik, Form, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import { fighterService, predictionService } from '../services/api';
import PredictionResult from '../components/prediction/PredictionResults';
import FighterSelector from '../components/fighters/FighterSelector';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { motion } from 'framer-motion';

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
  const [selectedFighter1, setSelectedFighter1] = useState(null);
  const [selectedFighter2, setSelectedFighter2] = useState(null);

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
      
      // Set selected fighters for display
      setSelectedFighter1(fighter1Data);
      setSelectedFighter2(fighter2Data);
      
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

  // Animation variants
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Hero Section */}
      <div className="relative mb-12">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-gray-900 to-transparent h-48 -z-10"></div>
        
        <motion.h1 
          className="text-4xl font-extrabold pt-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-red-500 via-white to-blue-500"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          Predict UFC Fight Outcomes
        </motion.h1>
        
        <motion.p 
          className="text-lg text-gray-300 max-w-2xl mx-auto text-center font-light mt-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.7 }}
        >
          Our advanced AI analyzes fighter statistics to forecast who will emerge victorious
        </motion.p>
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner text="Loading fighter data..." />
        </div>
      ) : error && !predictionResult ? (
        <ErrorAlert message={error} />
      ) : (
        <motion.div 
          className="grid grid-cols-1 lg:grid-cols-5 gap-8"
          variants={container}
          initial="hidden"
          animate="show"
        >
          {/* Fighter Selection Form - 2 columns */}
          <motion.div 
            className="lg:col-span-2 card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl"
            variants={item}
          >
            <h2 className="text-2xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-blue-400 inline-block">Select Fighters</h2>
            
            <Formik
              initialValues={{ fighter1: '', fighter2: '' }}
              validationSchema={validationSchema}
              onSubmit={handleSubmit}
            >
              {({ isSubmitting, values, setFieldValue }) => (
                <Form>
                  <div className="mb-6">
                    <label className="block text-gray-300 mb-2 font-medium">
                      Red Corner Fighter
                    </label>
                    <FighterSelector
                      name="fighter1"
                      fighters={fighters}
                      cornerClass="text-red-500 border-red-500/30 focus:border-red-500/70"
                    />
                    <ErrorMessage name="fighter1" component="div" className="text-red-500 text-sm mt-1 font-medium" />
                  </div>
                  
                  <div className="flex justify-center my-4">
                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="h-px w-16 bg-gray-700"></div>
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-gray-900 text-gray-400">VS</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mb-8">
                    <label className="block text-gray-300 mb-2 font-medium">
                      Blue Corner Fighter
                    </label>
                    <FighterSelector
                      name="fighter2"
                      fighters={fighters}
                      cornerClass="text-blue-500 border-blue-500/30 focus:border-blue-500/70"
                    />
                    <ErrorMessage name="fighter2" component="div" className="text-red-500 text-sm mt-1 font-medium" />
                  </div>
                  
                  <button
                    type="submit"
                    className="w-full py-3 btn-primary rounded-md font-bold tracking-wide transition-all duration-300 transform hover:scale-105 disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none"
                    disabled={isSubmitting || predicting || !values.fighter1 || !values.fighter2}
                  >
                    {predicting ? (
                      <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Analyzing Fight...
                      </span>
                    ) : 'Predict Fight Outcome'}
                  </button>
                </Form>
              )}
            </Formik>
          </motion.div>
          
          {/* Prediction Results - 3 columns */}
          <motion.div 
            className="lg:col-span-3"
            variants={item}
          >
            {predicting ? (
              <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl text-center py-12 h-full flex flex-col justify-center items-center">
                <div className="mb-6">
                  <div className="relative inline-flex">
                    <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center animate-pulse">
                      <span className="text-red-500">{selectedFighter1?.name?.charAt(0) || '?'}</span>
                    </div>
                    <div className="absolute top-0 right-0 -mr-6 w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center animate-pulse">
                      <span className="text-blue-500">{selectedFighter2?.name?.charAt(0) || '?'}</span>
                    </div>
                  </div>
                </div>
                <LoadingSpinner text="AI analyzing fight data..." />
                <p className="text-gray-400 mt-4 max-w-sm">
                  Our advanced machine learning model is calculating win probabilities and fight outcome...
                </p>
              </div>
            ) : predictionResult ? (
              <PredictionResult result={predictionResult} fighter1={selectedFighter1} fighter2={selectedFighter2} />
            ) : (
              <div className="card backdrop-blur-sm bg-gray-900/60 border border-gray-800 shadow-xl text-center py-12 h-full flex flex-col justify-center items-center">
                <h2 className="text-2xl font-bold mb-4">Fight Prediction</h2>
                <div className="text-7xl text-gray-700 mb-6">VS</div>
                <p className="text-gray-400 max-w-sm">
                  Select two fighters from the dropdown menus and click "Predict Fight Outcome" 
                  to see who our AI predicts will win.
                </p>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
      
      <motion.div 
        className="mt-12 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 rounded-xl shadow-xl p-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        <h3 className="text-xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400 inline-block">
          How Our AI Prediction Works
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="flex flex-col items-center text-center p-4 rounded-lg bg-gray-800/30">
            <div className="bg-purple-900/30 p-3 rounded-full mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h4 className="text-white font-semibold mb-2">Statistical Analysis</h4>
            <p className="text-gray-400 text-sm">
              We analyze fight statistics including striking accuracy, takedown defense, 
              and submission rates to identify performance patterns.
            </p>
          </div>
          
          <div className="flex flex-col items-center text-center p-4 rounded-lg bg-gray-800/30">
            <div className="bg-blue-900/30 p-3 rounded-full mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h4 className="text-white font-semibold mb-2">Style Matchup Analysis</h4>
            <p className="text-gray-400 text-sm">
              Our AI evaluates how different fighting styles match up against each other, 
              identifying advantages in striker vs grappler confrontations.
            </p>
          </div>
          
          <div className="flex flex-col items-center text-center p-4 rounded-lg bg-gray-800/30">
            <div className="bg-red-900/30 p-3 rounded-full mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h4 className="text-white font-semibold mb-2">Physical Advantages</h4>
            <p className="text-gray-400 text-sm">
              We factor in reach, height, and weight differences between fighters to assess
              physical advantages that may influence the outcome.
            </p>
          </div>
        </div>
        
        <p className="text-gray-400 mt-8 text-sm text-center max-w-3xl mx-auto">
          <span className="font-semibold text-yellow-400">Disclaimer:</span> Predictions are based on statistical models and historical data. 
          UFC fights are inherently unpredictable, and our model provides probabilities rather than guarantees.
          Always exercise responsible decision-making when using prediction data. Don't Gamble.
        </p>
      </motion.div>
    </motion.div>
  );
};

export default Prediction;