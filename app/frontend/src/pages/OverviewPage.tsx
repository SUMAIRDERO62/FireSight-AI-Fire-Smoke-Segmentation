import {
  Activity, Clock3, Database, Layers3, ScanLine, ShieldCheck
} from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from 'recharts'
import { motion } from 'motion/react'
import MetricCard from '../components/MetricCard'

const benchmarkData = [
  { metric: 'Dice score', unet: 82.9, yolo: 86.145 },
  { metric: 'IoU score', unet: 71.488, yolo: 75.827 },
  { metric: 'Precision', unet: 80.982, yolo: 82.977 },
  { metric: 'Recall', unet: 84.912, yolo: 89.605 },
]

export default function OverviewPage() {
  return (
    <motion.div
      className="page-content"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <section className="welcome-row">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> COMPUTER VISION WORKSPACE</div>
          <h1>Explore Fire & Smoke Intelligence<span>.</span></h1>
          <p className="welcome-copy">A computer vision workspace for students, educators and researchers to explore image-based fire and smoke segmentation.</p>
        </div>
        <div className="date-chip"><Activity size={15} /> Model workspace</div>
      </section>

      <section className="metric-grid">
        <MetricCard label="Dataset images" value="201" detail="Verified segmentation images" icon={Database} index={0} />
        <MetricCard label="Polygon annotations" value="479" detail="Fire and smoke regions" icon={Layers3} index={1} />
        <MetricCard label="Segmentation models" value="02" detail="U-Net and YOLO11-seg" icon={ScanLine} index={2} />
        <MetricCard label="Test images" value="20" detail="Shared Phase 05 test split" icon={ShieldCheck} index={3} />
      </section>

      <section className="content-grid">
        <article className="panel performance-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">MODEL EVALUATION</div>
              <h2>Performance comparison</h2>
              <p>Measured on the same held-out test set</p>
            </div>
            <div className="benchmark-tag"><ShieldCheck size={14} /> Phase 05 benchmark</div>
          </div>

          <div className="chart-legend">
            <span><i className="legend-dot navy" /> U-Net</span>
            <span><i className="legend-dot amber" /> YOLO11-seg</span>
            <span className="chart-unit">SCORE (%)</span>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmarkData} barGap={7} barCategoryGap="27%" margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#E9EDF2" vertical={false} />
                <XAxis dataKey="metric" tick={{ fill: '#738096', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#8994A5', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ fill: '#F4F6F9' }}
                  contentStyle={{ border: '1px solid #E6EAF0', borderRadius: 10, fontSize: 12 }}
                  formatter={(value) => [`${Number(value).toFixed(2)}%`]}
                />
                <Bar dataKey="unet" name="U-Net" fill="#142B4A" radius={[5, 5, 0, 0]} maxBarSize={25} />
                <Bar dataKey="yolo" name="YOLO11-seg" fill="#F59E0B" radius={[5, 5, 0, 0]} maxBarSize={25} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-footnote">Evaluation metrics are from the verified 20-image Phase 05 test split. They are not live predictions.</div>
        </article>

        <article className="panel insight-panel">
          <div className="eyebrow">QUICK INSIGHT</div>
          <h2>YOLO11-seg leads on this test split</h2>
          <p>It achieved a higher Dice score and IoU than U-Net in the Phase 05 comparison.</p>
          <div className="insight-score">
            <div>
              <span>YOLO11-seg Dice</span>
              <strong>86.15%</strong>
            </div>
            <div className="score-mark"><Activity size={20} /></div>
          </div>
          <div className="insight-divider" />
          <div className="insight-row"><span>U-Net Dice</span><strong>82.90%</strong></div>
          <div className="insight-row"><span>YOLO11-seg IoU</span><strong>75.83%</strong></div>
          <div className="insight-note"><Clock3 size={14} /> Small test set — results need broader validation.</div>
        </article>
      </section>

      <section className="bottom-grid">
        <article className="panel model-card">
          <div className="model-icon unet-icon"><Layers3 size={20} /></div>
          <div className="model-copy">
            <span className="model-kicker">SEGMENTATION MODEL 01</span>
            <h3>U-Net</h3>
            <p>Pixel-level fire and smoke segmentation</p>
          </div>
          <span className="model-type">PyTorch</span>
        </article>
        <article className="panel model-card">
          <div className="model-icon yolo-icon"><ScanLine size={20} /></div>
          <div className="model-copy">
            <span className="model-kicker">SEGMENTATION MODEL 02</span>
            <h3>YOLO11-seg</h3>
            <p>Instance segmentation for fire and smoke</p>
          </div>
          <span className="model-type">Ultralytics</span>
        </article>
      </section>
      <section className="learning-intro-grid">
        <article className="panel learning-intro-card">
          <div className="eyebrow">FOR STUDENTS</div>
          <h3>Learn computer vision by experimenting.</h3>
          <p>Explore segmentation models, upload images, inspect masks and understand evaluation metrics through practical experiments.</p>
        </article>
        <article className="panel learning-intro-card">
          <div className="eyebrow">FOR EDUCATORS</div>
          <h3>Explain models with visual evidence.</h3>
          <p>Use model outputs, architecture summaries and benchmark results as educational material for computer vision and deep learning.</p>
        </article>
        <article className="panel learning-intro-card">
          <div className="eyebrow">FOR RESEARCHERS</div>
          <h3>Study segmentation performance.</h3>
          <p>Compare model outputs and examine evaluation methodology, dataset limitations and opportunities for further validation.</p>
        </article>
      </section>
      <footer className="page-footer">FIRESIGHT AI <span>·</span> IMAGE-BASED FIRE & SMOKE SEGMENTATION</footer>
    </motion.div>
  )
}

