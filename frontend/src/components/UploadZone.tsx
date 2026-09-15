import { FileImage, Upload, X } from "lucide-react"
import { useRef, useState } from "react"

interface Props {
  label: string
  hint: string
  file: File | null
  onChange: (file: File | null) => void
  optional?: boolean
}

export function UploadZone({ label, hint, file, onChange, optional }: Props) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFiles(files: FileList | null) {
    if (files && files[0]) onChange(files[0])
  }

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        {optional && <span className="text-[11px] text-ink-faint">optional</span>}
      </div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          handleFiles(e.dataTransfer.files)
        }}
        className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-lg border px-4 py-5 text-center transition-colors ${
          dragging
            ? "border-brand-500 bg-brand-500/[0.06]"
            : file
              ? "border-border-soft bg-surface-raised"
              : "border-dashed border-border hover:border-border-soft hover:bg-surface-hover"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,image/tiff"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {file ? (
          <div className="flex w-full items-center gap-2.5">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-brand-500/10 text-brand-400">
              <FileImage size={16} />
            </div>
            <div className="min-w-0 flex-1 text-left">
              <div className="truncate text-[13px] text-ink">{file.name}</div>
              <div className="text-[11px] text-ink-faint">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onChange(null)
                if (inputRef.current) inputRef.current.value = ""
              }}
              className="shrink-0 rounded-md p-1 text-ink-faint hover:bg-surface-hover hover:text-ink"
            >
              <X size={14} />
            </button>
          </div>
        ) : (
          <>
            <Upload size={18} className="mb-2 text-ink-faint group-hover:text-ink-muted" />
            <div className="text-[13px] text-ink-muted">
              Drop GeoTIFF or <span className="text-brand-400">browse</span>
            </div>
            <div className="mt-0.5 text-[11px] text-ink-faint">{hint}</div>
          </>
        )}
      </div>
    </div>
  )
}
