import { ArrowUpRight, ExternalLink, ScanLine } from 'lucide-react'
import { motion } from 'motion/react'

export default function AboutPage() {
  return (
    <motion.div className="page-content feature-page" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <section className="feature-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> THE PROJECT</div>
          <h1>About FireSight AI<span>.</span></h1>
          <p>An image-based computer vision project for fire and smoke segmentation.</p>
        </div>
        <div className="feature-heading-icon"><ScanLine size={22} /></div>
      </section>

      <article className="panel about-hero">
        <div className="about-brand-mark"><ScanLine size={28} /></div>
        <div className="eyebrow">COMPUTER VISION / DEEP LEARNING</div>
        <h2>Seeing the details<br />behind fire & smoke.</h2>
        <p>FireSight AI explores pixel-level and instance segmentation through two trained models: U-Net and YOLO11-seg. The workspace provides image-based inference, visual mask inspection and model evaluation using a shared test split.</p>
        <div className="about-tags"><span>U-Net</span><span>YOLO11-seg</span><span>PyTorch</span><span>FastAPI</span><span>React</span></div>
      </article>

      <div className="about-profile-grid">
        <article className="panel about-profile">
          <div className="about-avatar">SD</div>
          <div className="eyebrow">DEVELOPER</div>
          <h2>Sumair Ahmed Dero</h2>
          <p>BS Artificial Intelligence student</p>
          <p className="profile-description">Interested in artificial intelligence, machine learning and practical computer vision applications.</p>
          <div className="profile-links">
            <a href="https://www.linkedin.com/in/sumair-ahmed-dero-70ba852a0/" target="_blank" rel="noreferrer">
              <ExternalLink size={17} /> LinkedIn profile <ExternalLink size={14} />
            </a>
            <a href="https://sumair-ahmed-dero-ai-journey.vercel.app/" target="_blank" rel="noreferrer">
              <ArrowUpRight size={17} /> AI Journey portfolio <ExternalLink size={14} />
            </a>
          </div>
        </article>

        <article className="panel about-project-info">
          <div className="eyebrow">PROJECT OVERVIEW</div>
          <h2>Built around real model outputs.</h2>
          <div className="about-info-row"><span>Project</span><strong>FireSight AI</strong></div>
          <div className="about-info-row"><span>Task</span><strong>Fire & smoke segmentation</strong></div>
          <div className="about-info-row"><span>Input</span><strong>Images only</strong></div>
          <div className="about-info-row"><span>Models</span><strong>U-Net / YOLO11-seg</strong></div>
          <div className="about-info-row"><span>Backend</span><strong>FastAPI</strong></div>
          <div className="about-info-row"><span>Frontend</span><strong>React + TypeScript</strong></div>
        </article>
      </div>
      <footer className="page-footer">FIRESIGHT AI <span>·</span> BUILT BY SUMAIR AHMED DERO</footer>
    </motion.div>
  )
}


