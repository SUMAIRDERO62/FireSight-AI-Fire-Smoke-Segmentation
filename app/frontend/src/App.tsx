import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ArrowUpRight, BarChart3, Camera } from 'lucide-react'
import Sidebar, { type PageName } from './components/Sidebar'
import Topbar from './components/Topbar'
import OverviewPage from './pages/OverviewPage'
import ImageAnalysisPage from './pages/ImageAnalysisPage'
import AIModelsPage from './pages/AIModelsPage'
import AboutPage from './pages/AboutPage'
import ModelComparisonPage from './pages/ModelComparisonPage'
import ResultPanel from './components/ResultPanel'
import { getHealth } from './services/api'
import type { HealthResponse, PredictionRecord } from './types/api'

export default function App() {
  const [activePage, setActivePage] = useState<PageName>('Overview')
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [latestResult, setLatestResult] = useState<PredictionRecord | null>(null)

  useEffect(() => {
    let active = true

    const checkApi = async () => {
      try {
        const data = await getHealth() as HealthResponse
        if (active) {
          setHealth(data)
          setApiOnline(true)
        }
      } catch {
        if (active) {
          setApiOnline(false)
          setHealth(null)
        }
      }
    }

    void checkApi()
    const interval = window.setInterval(() => void checkApi(), 15000)

    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  function renderPage() {
    switch (activePage) {
      case 'Overview':
        return <OverviewPage />
      case 'AI Models':
        return <AIModelsPage health={health} />
      case 'Image Analysis':
        return <ImageAnalysisPage latestResult={latestResult} onPredictionComplete={setLatestResult} />
      case 'Results':
        return latestResult ? (
          <div className="page-content feature-page">
            <section className="feature-heading">
              <div>
                <div className="eyebrow"><span className="eyebrow-line" /> SESSION OUTPUT</div>
                <h1>Results<span>.</span></h1>
                <p>Your latest completed image segmentation result.</p>
              </div>
              <div className="feature-heading-icon"><BarChart3 size={22} /></div>
            </section>
            <ResultPanel record={latestResult} />
          </div>
        ) : (
          <motion.div className="empty-results panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="empty-results-icon"><Camera size={25} /></div>
            <div className="eyebrow">NO SESSION OUTPUT</div>
            <h1>No results yet<span>.</span></h1>
            <p>Run an image segmentation from Image Analysis. Your latest result will appear here.</p>
            <button onClick={() => setActivePage('Image Analysis')}>Open image analysis <ArrowUpRight size={16} /></button>
          </motion.div>
        )
      case 'Model Comparison':
        return <ModelComparisonPage />
      case 'About':
        return <AboutPage />
      default:
        return <OverviewPage />
    }
  }

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <main className="main-area">
        <Topbar page={activePage} apiOnline={apiOnline} />
        <div className="main-scroll">
          <AnimatePresence mode="wait">
            <motion.div
              key={activePage}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              transition={{ duration: 0.18 }}
            >
              {renderPage()}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
      <div className="floating-shortcut"><ArrowUpRight size={15} /> FIRESIGHT AI</div>
    </div>
  )
}


