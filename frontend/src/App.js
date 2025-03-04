import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/navbar';
import Footer from './components/layout/footer';
import Home from './pages/Home';
import Fighters from './pages/Fighters';
import FighterDetail from './pages/FighterDetail';
import Prediction from './pages/Prediction';
import NotFound from './pages/NotFound';

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-dark-bg text-white">
        <Navbar />
        <main className="flex-grow container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/fighters" element={<Fighters />} />
            <Route path="/fighters/:name" element={<FighterDetail />} />
            <Route path="/predict" element={<Prediction />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;