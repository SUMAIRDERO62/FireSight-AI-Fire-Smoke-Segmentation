import { Bell, CircleHelp, Command } from 'lucide-react'

type TopbarProps = {
  page: string
  apiOnline: boolean | null
}

export default function Topbar({ page, apiOnline }: TopbarProps) {
  return (
    <header className="topbar">
      <div className="breadcrumb">
        <span>FireSight AI</span>
        <span className="breadcrumb-separator">/</span>
        <strong>{page}</strong>
      </div>
      <div className="topbar-actions">
        <div className={`api-status ${apiOnline === true ? 'online' : apiOnline === false ? 'offline' : ''}`}>
          <span className="status-dot" />
          {apiOnline === true ? 'API connected' : apiOnline === false ? 'API offline' : 'Checking API'}
        </div>
        <button className="icon-button" aria-label="Help"><CircleHelp size={18} /></button>
        <button className="icon-button" aria-label="Notifications"><Bell size={18} /></button>
        <div className="topbar-divider" />
        <div className="command-chip"><Command size={14} /> Workspace</div>
      </div>
    </header>
  )
}
