{{- define "v3.name" -}}
{{- required "Set workload in the environment values file" .Values.workload -}}
{{- end -}}
{{- define "v3.selectorLabels" -}}
app.kubernetes.io/name: {{ include "v3.name" . }}
app.kubernetes.io/instance: {{ include "v3.name" . }}
{{- end -}}
{{- define "v3.labels" -}}
{{ include "v3.selectorLabels" . }}
app.kubernetes.io/component: {{ .Values.component | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}
{{- define "v3.validate" -}}
{{- if ne .Release.Namespace .Values.environment -}}
{{- fail "Release namespace must equal environment (staging or prod)" -}}
{{- end -}}
{{- if ne .Values.nodeSelector.environment .Values.environment -}}
{{- fail "nodeSelector.environment must equal environment" -}}
{{- end -}}
{{- if and .Values.hpa.enabled (lt (int .Values.hpa.maxReplicas) (int .Values.hpa.minReplicas)) -}}
{{- fail "HPA maxReplicas must be at least minReplicas" -}}
{{- end -}}
{{- if and .Values.hpa.enabled (not .Values.service.enabled) -}}
{{- fail "HPA is only enabled for HTTP API workloads" -}}
{{- end -}}
{{- if and (hasKey .Values.resources.requests "nvidia.com/gpu") (ne .Values.environment "prod") -}}
{{- fail "GPU resources are only allowed in prod" -}}
{{- end -}}
{{- if and (eq .Values.workload "embedding-service") (eq .Values.environment "prod") -}}
{{- if or (ne .Values.env.EMBEDDING_DEVICE "cuda") (not (hasSuffix "-gpu" .Values.image.tag)) -}}
{{- fail "prod embedding requires EMBEDDING_DEVICE=cuda and a SHA-gpu image tag" -}}
{{- end -}}
{{- if or (ne (int (index .Values.resources.requests "nvidia.com/gpu")) 1) (ne (int (index .Values.resources.limits "nvidia.com/gpu")) 1) -}}
{{- fail "prod embedding requires one GPU in both requests and limits" -}}
{{- end -}}
{{- end -}}
{{- end -}}
