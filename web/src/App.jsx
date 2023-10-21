import LoadingPage from './pages/LoadingPage';
import { useEffect, useState } from 'react';

export default function App() {
  const [loading, setLoading] = useState(0);

  useEffect(() => {
      const loadData = async () => {
        for (let i = 0; i <= 100; i++) { // Use `<=` to include 100%.
          await new Promise((resolve) =>
            setTimeout(() => {
              setLoading(i);
              resolve();
            }, 20)
          );
        }
      };
  
      loadData(); 
    }, []);

  return (
    <>
    {loading == 100 ? (<div>Loaded</div>) : (<LoadingPage loading={loading}></LoadingPage>)}
    </>
  );
}

