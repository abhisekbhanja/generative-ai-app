import { useState } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [files, setFiles] = useState([])
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [selectedFileCount, setSelectedFileCount] = useState(0)

  const handleFileSelection = (event) => {
    const selectedFiles = Array.from(event.target.files || [])
    setFiles(selectedFiles)
    setSelectedFileCount(selectedFiles.length)
    setUploadStatus('')
    setUploadSuccess(false)
    setAnswer('')
  }

  const handleUpload = async () => {
    if (!files.length) {
      setUploadStatus('No file selected.')
      return
    }

    setLoading(true)
    setAnswer('')
    setUploadStatus('Uploading files...')

    try {
      const results = []

      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        })

        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.detail || 'File upload failed')
        }

        results.push(`${file.name} → indexed (${data.chunks} chunks)`)
      }

      setUploadStatus(`file uploaded successfully in vector database`)
      setTimeout(() => {
        setUploadStatus('')
      }, 5000);
      setUploadSuccess(true)
      //setAnswer('Files are now available in the vector database.')
    } catch (error) {
      setUploadStatus(error.message || 'Unable to upload files to the backend.')
      setUploadSuccess(false)
      setAnswer('Upload failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!question.trim()) return

    setLoading(true)
    setAnswer('')

    try {
      const response = await fetch(`${API_BASE_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Request failed')
      }
      console.log(data);
      
      setAnswer(data)
    } catch (error) {
      setAnswer(error.message || 'Unable to reach the backend server.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <div className="card">
        <h1>RAG File Upload & Query</h1>
        <p>Upload documents and ask questions in a simple UI.</p>

        <label className="upload-box">
          <span>Choose files</span>
          <input type="file" multiple onChange={handleFileSelection} />
        </label>
        <br /><br />
        <button type="button" onClick={handleUpload} disabled={loading}>
          {loading ? 'Uploading...' : 'Upload to Vector DB'}
        </button>
{uploadStatus ? (
            <p className={uploadSuccess ? 'success' : 'error'}>{uploadStatus}</p>
          ) : null}
        <div className="file-list">
          {selectedFileCount > 0 ? (
            files.map((file) => <div key={file.name}>{file.name}</div>)
          ) : (
            <p className="muted">No files selected yet.</p>
          )}
        </div>

        <form onSubmit={handleSubmit} className="input-row">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your files..."
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>
        <div className="answer-box">
          
          {answer ? <p>{answer}</p> : null}
        </div>
      </div>
    </div>
  )
}

export default App
