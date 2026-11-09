#!/usr/bin/env bash
set -euo pipefail
helm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 검증용 태그는 렌더링에만 사용한다. values 파일이나 ECR 이미지를 변경하지 않는다.
for environment in staging prod; do
  for values_file in "$helm_root/values/$environment/"*.yaml; do
    workload="$(basename "$values_file" .yaml)"
    case "$workload" in
      resident-api|manager-api|chat-consumer) chart=backend-workload ;;
      *) chart="$workload" ;;
    esac
    tag=abcdef0
    if [[ "$workload" == embedding-service && "$environment" == prod ]]; then tag=abcdef0-gpu; fi
    helm lint "$helm_root/charts/$chart" --strict --namespace "$environment" -f "$values_file" --set-string image.tag="$tag"
    helm template "$environment-$workload" "$helm_root/charts/$chart" --namespace "$environment" --kube-version 1.36.0 -f "$values_file" --set-string image.tag="$tag" >/dev/null
    printf 'PASS %s/%s\n' "$environment" "$workload"
  done
done
