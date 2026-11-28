import { Activity, CheckCircle2, Layers3, ScanLine, ShieldCheck } from 'lucide-react'
import { motion } from 'motion/react'
import type { HealthResponse } from '../types/api'

type Props = { health: HealthResponse | null }

const models = [
  {
    name: 'U-Net',
    key: 'unet',
    framework: 'PyTorch',
    description: 'Encoder-decoder convolutional architecture for pixel-level fire and smoke segmentation.',
    dice: '82.90%',
    iou: '71.49%',
    precision: '80.98%',
    recall: '84.91%',
    time: '0.02417 s',
    size: '23.422 MB',
  },
  {
    name: 'YOLO11-seg',
    key: 'yolo11-seg',
    framework: 'Ultralytics',
    description: 'Instance segmentation model for identifying fire and smoke regions.',
    dice: '86.15%',
    iou: '75.83%',
    precision: '82.98%',
    recall: '89.61%',
    time: '0.10832 s',
    size: '6.047 MB',
  },
]

export default function AIModelsPage({ health }: Props) {
  return (
    <motion.div className="page-content feature-page" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <section className="feature-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> MODEL REGISTRY</div>
          <h1>AI models<span>.</span></h1>
          <p>Explore the trained segmentation models and their verified benchmark results.</p>
        </div>
        <div className="feature-heading-icon"><Layers3 size={22} /></div>
      </section>

      <div className="model-registry-grid">
        {models.map((model, index) => {
          const status = health?.models?.[model.key]
          return (
            <article className="panel registry-card" key={model.key}>
              <div className="registry-card-top">
                <div className={`registry-model-icon ${index === 1 ? 'amber-option' : ''}`}>
                  {index === 0 ? <Layers3 size={22} /> : <ScanLine size={22} />}
                </div>
                <span className={`checkpoint-status ${status?.checkpoint_available ? 'available' : 'unavailable'}`}>
                  <span /> {status ? (status.checkpoint_available ? 'Checkpoint available' : 'Checkpoint missing') : 'Checking status'}
                </span>
              </div>
              <div className="eyebrow">SEGMENTATION MODEL 0{index + 1}</div>
              <h2>{model.name}</h2>
              <p>{model.description}</p>
              <div className="framework-label">{model.framework}</div>
              <div className="registry-metrics">
                <div><span>Dice</span><strong>{model.dice}</strong></div>
                <div><span>IoU</span><strong>{model.iou}</strong></div>
                <div><span>Precision</span><strong>{model.precision}</strong></div>
                <div><span>Recall</span><strong>{model.recall}</strong></div>
              </div>
              <div className="registry-meta">
                <span><Activity size={14} /> {model.time} / image</span>
                <span><ShieldCheck size={14} /> {model.size}</span>
              </div>
            </article>
          )
        })}
      </div>

      <article className="panel benchmark-context">
        <div className="eyebrow">EVALUATION CONTEXT</div>
        <h2>Shared Phase 05 test split</h2>
        <div className="benchmark-facts">
          <div><strong>20</strong><span>Test images</span></div>
          <div><strong>201</strong><span>Dataset images</span></div>
          <div><strong>479</strong><span>Polygon annotations</span></div>
        </div>
        <p><CheckCircle2 size={15} /> Both models were evaluated on the same held-out test split. This is a small test set, so broader validation is needed before drawing general conclusions.</p>
        <p className="benchmark-footnote">Inference times and checkpoint sizes are recorded Phase 05 benchmark values, not live API request timings.</p>
      </article>
    </motion.div>
  )
}
