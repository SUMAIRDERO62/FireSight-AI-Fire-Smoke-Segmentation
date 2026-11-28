import type { LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'

type MetricCardProps = {
  label: string
  value: string
  detail: string
  icon: LucideIcon
  index: number
}

export default function MetricCard({ label, value, detail, icon: Icon, index }: MetricCardProps) {
  return (
    <motion.article
      className="metric-card"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
      whileHover={{ y: -3 }}
    >
      <div className="metric-top">
        <span>{label}</span>
        <div className="metric-icon"><Icon size={18} /></div>
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </motion.article>
  )
}
