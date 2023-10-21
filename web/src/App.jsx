import LoadingPage from './pages/LoadingPage';
import HomePage from './pages/HomePage';
import { useEffect, useState } from 'react';

export default function App() {
  const [loading, setLoading] = useState(0);

  const anim = (x) => {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  useEffect(() => {
      const loadData = async () => {
        for (let i = 0; i <= 100; i++) { // Use `<=` to include 100%.
          await new Promise((resolve) =>
            setTimeout(() => {
              setLoading(anim((i/100)) * 100);
              resolve();
            }, 1)
          );
        }
      };
  
      loadData(); 
    }, []);

  return (
    <>
      {loading == 100 ? (<HomePage></HomePage>) : (<LoadingPage loading={loading}></LoadingPage>)}
    </>
  );
}

